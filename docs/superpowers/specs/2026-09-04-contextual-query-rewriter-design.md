# Contextual Query Rewriter Design

## Goal

Preserve the semantic intent of elliptical follow-up questions without allowing
conversation history to become paper evidence or to change the resolved paper scope.

## Design

`ContextualQueryRewriter` runs after `TurnContextBuilder`. It receives the current
question, resolved paper IDs and titles, active section/task, bounded recent
dialogue, and the rolling conversation summary. For contextual research turns it makes at
most one LLM call and returns a standalone `retrieval_query`. If no useful history
exists, no LLM is configured, parsing fails, or the provider is unavailable, it
returns the original question.

The component cannot mutate `paper_context`, `paper_ids`, or external-search
admission. Retriever uses `retrieval_query` for MultiQuery, embeddings, graph
candidate lookup, and section detection. DirectAnalyzer uses the same query for
section selection and its analysis prompt so single-paper follow-ups behave
consistently. User-memory recording continues to use the original message.

Recent assistant answers may identify referents such as “the three answers” or
“your second point”, but their claims are not treated as paper facts. Conversation
fields are prompt context only. They are never appended to
`retrieved_chunks`, full paper text, citations, or Analyzer evidence.

## Data and workflow

The state adds `retrieval_query` and `recent_user_messages`. Web state creation
collects a bounded tail of prior dialogue and excludes the current turn. The
workflow becomes `TurnContext -> ContextualQueryRewriter -> existing intent route`.
Search and general turns pass through unchanged. Critic retries reuse the existing
query and do not trigger another rewrite.

## Verification

Focused tests cover contextual rewriting, no-context passthrough, LLM failure
fallback, web history construction, Retriever query consumption, and
DirectAnalyzer query consumption.
