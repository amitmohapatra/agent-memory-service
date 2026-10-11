"""Task names shared by the modules that enqueue a job and the registry that runs it (plain
assignments: the registration check reads them as module-level string constants)."""

TASK_MEMORY_INDEX = "memory.index"
#: Graph enrichment of indexed memories / of an indexed document version: its own job on the
#: GRAPH queue, one at a time per tenant, so the LLM calls it makes never hold back indexing
TASK_GRAPH_ENRICH_MEMORIES = "graph.enrich_memories"
TASK_GRAPH_ENRICH_DOCUMENT = "graph.enrich_document"
