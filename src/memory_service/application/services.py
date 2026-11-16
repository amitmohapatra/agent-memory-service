"""The services of one process, by name and type.

``wire_adapters`` builds them in stages, so a service is an attribute that exists once its
stage has run: reading one that was not wired raises ``AttributeError``, and a caller for
which a service is optional asks ``get``. Reading and writing by name
(``services["memory"]``) is kept for the callers written against the dictionary this
replaced; an unknown name is refused where it is written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from memory_service.application.container import Container
    from memory_service.modules.agent_tools.service import AgentTools
    from memory_service.modules.archive.service import ArchiveService
    from memory_service.modules.audit.service import ReadAudit
    from memory_service.modules.auth.authentication import ServiceAuthenticator
    from memory_service.modules.auth.keys import ApiKeyVerifier
    from memory_service.modules.authz.service import AuthorizationService
    from memory_service.modules.context.builder import ContextBuilder
    from memory_service.modules.context.expansion import ExpansionStage
    from memory_service.modules.context.handles import BundleRecords
    from memory_service.modules.conversation.service import ConversationService
    from memory_service.modules.conversation.summary import ThreadSummaries
    from memory_service.modules.feedback.service import FeedbackService
    from memory_service.modules.graph.service import GraphService
    from memory_service.modules.grounding.cascade import GroundingCascade
    from memory_service.modules.idempotency.service import IdempotencyService
    from memory_service.modules.ingestion.service import IngestionService
    from memory_service.modules.llm.assist import LLMAssist
    from memory_service.modules.llm.credentials import ModelCredentials
    from memory_service.modules.llm.endpoints import EndpointResolver
    from memory_service.modules.llm.policies import LLMUsage, ModelPolicies
    from memory_service.modules.memory.connections import ConnectionService
    from memory_service.modules.memory.consolidation import ConsolidationService
    from memory_service.modules.memory.ephemeral import EphemeralMemory
    from memory_service.modules.memory.forgetting import ForgettingService
    from memory_service.modules.memory.pipeline import ObservationPipeline
    from memory_service.modules.memory.reflection import ReflectionService
    from memory_service.modules.memory.service import MemoryService
    from memory_service.modules.profile.service import ProfileService
    from memory_service.modules.rag.indexer import Indexer
    from memory_service.modules.retrieval.engine import RetrievalEngine
    from memory_service.modules.retrieval.search import Searcher
    from memory_service.modules.tenancy.registry import TenantRegistry
    from memory_service.modules.tenancy.retention import RetentionService
    from memory_service.modules.tenancy.service import TenancyService
    from memory_service.modules.tools.hints import ToolHintsService
    from memory_service.modules.tools.index import ToolIndex
    from memory_service.modules.tools.learning import ToolLearning
    from memory_service.modules.tools.permissions import ToolPermissions
    from memory_service.modules.tools.service import ToolMemoryService
    from memory_service.modules.tools.skills import LearnedSkills
    from memory_service.modules.working_memory.hot_thread import HotThreadCache
    from memory_service.ports.credentials import EndpointProbe
    from memory_service.ports.intelligence import MemoryIntelligenceProvider
    from memory_service.ports.uow import OutboxSweeper, UnitOfWorkFactory


class Services:
    # persistence
    uow_factory: UnitOfWorkFactory
    outbox_relay: OutboxSweeper
    idempotency: IdempotencyService
    # callers and tenants
    tenant_registry: TenantRegistry
    api_keys: ApiKeyVerifier
    authenticator: ServiceAuthenticator
    authz: AuthorizationService
    tenancy: TenancyService
    retention: RetentionService
    read_audit: ReadAudit
    # conversations and documents
    hot_thread: HotThreadCache
    conversation: ConversationService
    thread_summaries: ThreadSummaries
    profile: ProfileService
    archive_service: ArchiveService
    ingestion: IngestionService
    # models
    model_credentials: ModelCredentials
    model_policies: ModelPolicies
    llm_usage: LLMUsage
    llm_endpoints: EndpointResolver
    endpoint_probe: EndpointProbe
    llm_assist: LLMAssist
    grounding: GroundingCascade
    # retrieval and context
    indexer: Indexer
    retrieval: RetrievalEngine
    ephemeral_memory: EphemeralMemory
    context_builder: ContextBuilder
    bundle_records: BundleRecords
    search: Searcher
    expansion: ExpansionStage
    graph: GraphService
    # memories
    memory_provider: MemoryIntelligenceProvider
    observation_pipeline: ObservationPipeline
    memory: MemoryService
    feedback: FeedbackService
    forgetting: ForgettingService
    reflection: ReflectionService
    consolidation: ConsolidationService
    connections: ConnectionService
    # tools
    tool_memory: ToolMemoryService
    tool_index: ToolIndex
    tool_hints: ToolHintsService
    tool_learning: ToolLearning
    agent_tools: AgentTools
    learned_skills: LearnedSkills
    tool_permissions: ToolPermissions
    #: more work for the reconcile job, and more task registrations: for a build that adds
    #: jobs of its own (none is wired here)
    extra_reconcilers: list[Callable[[], Awaitable[None]]]
    extra_task_registrars: list[Callable[[Container], None]]

    def __init__(self, **services: Any) -> None:
        self.update(**services)

    @classmethod
    def names(cls) -> frozenset[str]:
        """The services this class declares."""
        return frozenset(cls.__annotations__)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.names() and hasattr(self, name)

    def __getitem__(self, name: str) -> Any:
        if name not in self:
            raise KeyError(name)
        return getattr(self, name)

    def __setitem__(self, name: str, service: Any) -> None:
        if name not in self.names():
            raise KeyError(f"{name!r} is not a service (application/services.py declares them)")
        setattr(self, name, service)

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default) if name in self.names() else default

    def update(self, **services: Any) -> None:
        for name, service in services.items():
            self[name] = service
