"""OpenFGA AuthorizationProvider (Apache-2.0, Zanzibar-inspired ReBAC).

The store and authorization model are created on first use when not configured, so the
dev stack needs no manual setup. Decisions are cached by (store, model, tuple-revision)
in the CacheProvider when available.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from openfga_sdk import ClientConfiguration, OpenFgaClient
from openfga_sdk.client.models import (
    ClientBatchCheckItem,
    ClientBatchCheckRequest,
    ClientCheckRequest,
    ClientListObjectsRequest,
    ClientTuple,
    ClientWriteRequest,
)
from openfga_sdk.credentials import CredentialConfiguration, Credentials
from openfga_sdk.models import CreateStoreRequest, WriteAuthorizationModelRequest
from openfga_sdk.sync import (
    OpenFgaClient as SyncOpenFgaClient,  # noqa: F401 - documents sync availability
)

from memory_service.config.settings import AuthorizationSettings
from memory_service.domain.errors import DependencyUnavailable
from memory_service.observability.logging import get_logger
from memory_service.observability.metrics import authz_denials_total, stage_seconds
from memory_service.observability.tracing import span
from memory_service.ports.authorization import AccessCheck, RelationTuple
from memory_service.ports.models import ProviderInfo

log = get_logger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[4] / "deploy" / "openfga" / "model.fga"
STORE_NAME = "memory-service"


class OpenFGAAuthorizationProvider:
    info = ProviderInfo(
        name="openfga",
        license="Apache-2.0",
        origin="openfga/openfga",
        locality="local",
        data_residency="deployment",
    )

    def __init__(self, settings: AuthorizationSettings, *, model_json: dict | None = None) -> None:
        self.settings = settings
        self._client: OpenFgaClient | None = None
        self._model_json = model_json
        self.max_listed_objects = settings.max_listed_objects

    async def _get_client(self) -> OpenFgaClient:
        if self._client is not None:
            return self._client
        credentials = None
        if self.settings.openfga_api_token is not None:
            credentials = Credentials(
                method="api_token",
                configuration=CredentialConfiguration(
                    api_token=self.settings.openfga_api_token.get_secret_value()
                ),
            )
        config = ClientConfiguration(
            api_url=self.settings.openfga_api_url,
            store_id=self.settings.openfga_store_id,
            authorization_model_id=self.settings.openfga_model_id,
            credentials=credentials,
            timeout_millisec=3000,
        )
        client = OpenFgaClient(config)
        try:
            if not config.store_id:
                config.store_id = await self._ensure_store(client)
                client.set_store_id(config.store_id)
            if not config.authorization_model_id:
                config.authorization_model_id = await self._ensure_model(client)
                client.set_authorization_model_id(config.authorization_model_id)
        except Exception as exc:
            # The client owns an aiohttp session. Without this, every failed attempt leaked
            # one — and since self._client is only set on success, a readiness probe polling
            # a broken OpenFGA leaked a session per poll.
            await client.close()
            if isinstance(exc, FileNotFoundError):
                raise DependencyUnavailable(
                    f"the OpenFGA authorization model is missing from this deployment "
                    f"({MODEL_PATH}); it is deploy/openfga/model.fga in the repository and "
                    f"must be present in the image"
                ) from exc
            raise DependencyUnavailable(f"OpenFGA unavailable: {type(exc).__name__}") from exc
        self._client = client
        return client

    async def _ensure_store(self, client: OpenFgaClient) -> str:
        stores = await client.list_stores()
        for store in stores.stores or []:
            if store.name == STORE_NAME:
                return store.id
        created = await client.create_store(CreateStoreRequest(name=STORE_NAME))
        return created.id

    async def _ensure_model(self, client: OpenFgaClient) -> str:
        models = await client.read_authorization_models()
        if models.authorization_models:
            return models.authorization_models[0].id
        body = self._model_json or _dsl_to_json(MODEL_PATH.read_text(encoding="utf-8"))
        written = await client.write_authorization_model(WriteAuthorizationModelRequest(**body))
        return written.authorization_model_id

    # -- port ------------------------------------------------------------------
    async def check(self, check: AccessCheck) -> bool:
        client = await self._get_client()
        with span("authz.check", relation=check.relation), stage_seconds.labels("authz").time():
            try:
                response = await client.check(
                    ClientCheckRequest(
                        user=check.user, relation=check.relation, object=check.object
                    )
                )
            except Exception as exc:
                raise DependencyUnavailable(f"OpenFGA check failed: {type(exc).__name__}") from exc
        allowed = bool(response.allowed)
        if not allowed:
            authz_denials_total.labels(check.relation).inc()
        return allowed

    async def batch_check(self, checks: Sequence[AccessCheck]) -> list[bool]:
        if not checks:
            return []
        client = await self._get_client()
        items = [
            ClientBatchCheckItem(
                user=c.user, relation=c.relation, object=c.object, correlation_id=str(i)
            )
            for i, c in enumerate(checks)
        ]
        try:
            response = await client.batch_check(ClientBatchCheckRequest(checks=items))
        except Exception as exc:
            raise DependencyUnavailable(
                f"OpenFGA batch_check failed: {type(exc).__name__}"
            ) from exc
        by_id = {r.correlation_id: bool(r.allowed) for r in response.result}
        return [by_id.get(str(i), False) for i in range(len(checks))]

    async def write(
        self, add: Sequence[RelationTuple], delete: Sequence[RelationTuple] = ()
    ) -> None:
        client = await self._get_client()
        body = ClientWriteRequest(
            writes=[ClientTuple(user=t.user, relation=t.relation, object=t.object) for t in add]
            or None,
            deletes=[ClientTuple(user=t.user, relation=t.relation, object=t.object) for t in delete]
            or None,
        )
        try:
            await client.write(body)
        except Exception as exc:
            raise DependencyUnavailable(f"OpenFGA write failed: {type(exc).__name__}") from exc

    async def list_objects(self, user: str, relation: str, object_type: str) -> list[str]:
        client = await self._get_client()
        try:
            response = await client.list_objects(
                ClientListObjectsRequest(user=user, relation=relation, type=object_type)
            )
        except Exception as exc:
            raise DependencyUnavailable(
                f"OpenFGA list_objects failed: {type(exc).__name__}"
            ) from exc
        return list(response.objects or [])[: self.max_listed_objects + 1]

    async def ping(self) -> bool:
        try:
            client = await self._get_client()
            await client.read_authorization_models()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


def _dsl_to_json(dsl: str) -> dict:
    """Convert the ``.fga`` DSL to the JSON authorization model accepted by the write API."""
    from memory_service.adapters.authz.model import parse_fga

    types = parse_fga(dsl)
    type_definitions = []
    for type_name, type_def in types.items():
        relations: dict = {}
        metadata_relations: dict = {}
        for rel_name, rel in type_def.relations.items():
            children = []
            if rel.direct:
                children.append({"this": {}})
            for computed in rel.computed:
                children.append({"computedUserset": {"object": "", "relation": computed}})
            for tupleset, target in rel.ttu:
                children.append(
                    {
                        "tupleToUserset": {
                            "tupleset": {"object": "", "relation": tupleset},
                            "computedUserset": {"object": "", "relation": target},
                        }
                    }
                )
            relations[rel_name] = (
                children[0] if len(children) == 1 else {"union": {"child": children}}
            )
            metadata_relations[rel_name] = {
                "directly_related_user_types": [
                    {"type": d.split("#")[0], "relation": d.split("#")[1]}
                    if "#" in d
                    else {"type": d}
                    for d in rel.direct
                ]
            }
        entry: dict = {"type": type_name, "relations": relations}
        if relations:
            entry["metadata"] = {"relations": metadata_relations}
        type_definitions.append(entry)
    return {"schema_version": "1.1", "type_definitions": type_definitions}
