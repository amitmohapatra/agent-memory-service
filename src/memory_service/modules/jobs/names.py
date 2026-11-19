"""Task names shared by the modules that enqueue a job and the registry that runs it (plain
assignments: the registration check reads them as module-level string constants)."""

TASK_MEMORY_INDEX = "memory.index"
#: A recorded message turned into memories; a message staged for the archive
TASK_PROCESS_OBSERVATION = "memory.process_observation"
TASK_ARCHIVE_STAGE = "archive.stage_message"
#: The periodic sweeps of memories past their expiry, and of what the forgetting policy drops
TASK_MEMORY_EXPIRE = "memory.expire"
TASK_MEMORY_FORGET = "memory.forget"
#: A document's chunks embedded and searchable, or taken out of the index once it is gone
#: (``Indexer.index_document``): ingestion enqueues it, and so does a thread's deletion
TASK_DOCUMENT_INDEX = "document.index"
#: A kept document's passages and graph facts moved to its document's audience (a thread
#: that held it was deleted, and live threads still hold it): a thread's deletion enqueues it
TASK_DOCUMENT_MOVE = "document.move"
#: Graph enrichment of indexed memories / of an indexed document version: its own job on the
#: GRAPH queue, in a pool of its own, so the LLM calls it makes never hold back indexing
TASK_GRAPH_ENRICH_MEMORIES = "graph.enrich_memories"
TASK_GRAPH_ENRICH_DOCUMENT = "graph.enrich_document"
#: One person's new facts folded into their beliefs (``modules.memory.consolidation``), and
#: the beliefs it changed indexed for recall (``Indexer.index_beliefs``)
TASK_CONSOLIDATE = "memory.consolidate"
TASK_BELIEF_INDEX = "belief.index"
