# RAG interface design

The interface is a daily document-search workspace. The primary task is to ask a question, read supporting evidence, and inspect its source. The workspace opens directly into that task.

## Design direction

An editorial knowledge workspace: warm ivory surfaces, a quiet stone sidebar, forest-green actions, serif question headings, system sans-serif interface text, and monospaced metadata. Numbered evidence references connect answers to the document library. Spacing, type sizes, borders, and state colors use shared CSS tokens.

The layout prioritizes the question and answer. A compact navigation sidebar holds conversations and the knowledge library. Source excerpts occupy a secondary evidence area. The sidebar becomes a drawer on narrow screens; document inspection uses a native dialog with keyboard dismissal and focus handling.

## Working interactions

The browser requests real state from the same FastAPI server. Health, document counts, answer excerpts, references, and elapsed time come from the backend. Saved conversations are browser-local and use separate IDs. The local retrieval engine answers each question independently; the interface does not claim model memory in local mode.

Loading disables repeated submission, errors offer a retry path, and unsupported questions get a no-evidence state. Copy controls, keyboard submission, library search, scope selection, and document inspection operate on real data. API-provided text is inserted as text, rather than interpreted as HTML.

Local mode is labeled as extractive search. It quotes source text instead of claiming to produce a language-model synthesis. Cloud generation requires explicitly configured provider services. Local scope controls and top-k apply to local retrieval; the original remote cloud workflow has its own retrieval settings. The browsable library contains bundled extracted documents, even when cloud generation is selected.

## Accessibility and responsive behavior

- Semantic headings, navigation, buttons, labeled fields, and a skip link.
- Visible keyboard focus and native dialog focus containment.
- Text labels alongside connection states and source types.
- Reduced-motion support and no artificial typewriter delay.
- Responsive constraints checked at small-phone, tablet, and desktop widths.
- No external font, icon, JavaScript, or CSS CDN required.

## Hive skills applied

The design follows the installed Hive library's Frontend UI Engineering and UI/UX Pro Max workflows, plus the verified Frontend Design Direction reference. The UI/UX search workflow was used for a design-system recommendation and accessibility/loading guidance. Marketing-layout suggestions were adapted to the existing document-search application.

- Frontend engineering: semantic controls, scoped state, real loading/error/empty states, accessible and responsive layout.
- UI/UX Pro Max: contrast, keyboard navigation, touch targets, motion preferences, readable typography, and clear state feedback.
- Frontend design direction: usable first screen, restrained product-specific styling, shared tokens, and deliberate hierarchy.
