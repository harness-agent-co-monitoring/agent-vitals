# Declaring sections

Vitals reports reading per **section**: a name and the exact text it covers in the context the model sees.
Only the harness knows which spans of its context are the task, the rules, the examples or the output format,
so the harness declares them. Read maps have one row per section and turn. Text that no section covers
still counts toward the context, so read lift compares each section with all of it.

## Automatic sections

When a harness declares nothing, `vitals serve` and `vitals replay` section each request
themselves (`vitals.core.auto_sections`), with names that stay the same across turns and episodes:

| Part | Name |
|---|---|
| system prompt | `system/<heading path>` per Markdown heading block (`system/preamble` before the first heading); a long prompt without headings is split by paragraph |
| tool schemas | `tool:<name>` |
| user turns | `user:1`, `user:2`, … |
| earlier assistant messages | `self:1`, `self:2`, … |
| tool results | `tool_result:<tool>#<k>`, the k-th result of that tool (tool output sent as a user message is `tool_result:output#<k>`) |

Declared sections always win: a message or tool list that holds a declared section gets no automatic ones, and
replay adds them only to turns the store has no declarations for. Automatic sections are marked `auto` in
`regions.jsonl`. `--no-auto-sections` turns them off.

## Ways to declare

| Your harness | Declare sections with |
|---|---|
| A static prompt file (system prompt, `AGENTS.md`, policy document) | `vitals annotate` (below), then `--spans-file` |
| A local agent served through `vitals serve` | `vitals.spans` in each chat-completions request, or `--spans-file` for all requests |
| Logs replayed offline | `--spans-file` for `vitals replay`, or `spans=` for `replay_episode` |
| A Meta-Harness candidate | `prompt_sections()` and `feedback_sections()` ([Meta-Harness contract](../vitals/addons/metaharness/SECTION_ANNOTATION.md)) |
| A GEPA program | nothing: every candidate component is declared automatically |

All of these take the same list:

```json
[
  {"name": "task", "text": "You are a coding agent working in /repo."},
  {"name": "rules", "text": "Never edit files outside /repo. Run the tests before finishing."},
  {"name": "feedback/checklist", "text": "Before finishing, confirm the tests pass.", "msg_index": 4}
]
```

## Static harnesses

A static harness is fixed text with no code that could declare its parts. `vitals annotate` derives the
declaration from the file itself.

```bash
vitals annotate system_prompt.md --by heading > sections.json      # one section per heading block
vitals annotate system_prompt.md --by paragraph > sections.json    # one section per paragraph
vitals annotate marked_prompt.md --by marker --clean-out system_prompt.md > sections.json
vitals annotate instance_template.md --by paragraph --template > sections.json
```

- **Headings.** Each Markdown heading and the text up to the next heading form a section, named by its heading
  path (`book_flight/payment`). Text before the first heading is `preamble`. A single leading `#` title is the
  document's own section and is left out of the other names. Headings inside fenced code are ignored.
- **Paragraphs.** Each blank-line separated block is a section, named by its heading path and first words
  (`rules/never_share_another_customer`). Use this when one heading holds several distinct rules. Repeated names
  get a `#2` suffix.
- **Markers.** For text without useful structure, mark sections inline. `<!-- section: name -->` opens a section,
  which runs to `<!-- /section -->`, the next marker, or the end of the file. Serve the cleaned file that
  `--clean-out` writes, since the markers must not reach the model.
- **Templates.** If the file is a prompt template with placeholders (`{{ task }}`, `{% if ... %}`, `{name}`,
  `${name}`), add `--template`. Each section is split at its placeholders and only the fixed text around them is
  declared, so the declaration matches every rendered instance. Fragments shorter than 8 characters are dropped.

The same functions are available from Python:

```python
from vitals.core.annotate import sections_from_markdown, sections_from_markers

sections = sections_from_markdown(open("system_prompt.md").read(), by="paragraph")
sections = sections_from_markdown(open("instance_template.md").read(), template=True)
clean_prompt, sections = sections_from_markers(open("marked_prompt.md").read())
```

Static files rarely cover everything the model sees. Tool results, retrieved documents and earlier turns are
added at run time. Declare harness text that is injected at run time with a `msg_index` anchor (below), or in
Meta-Harness with `feedback_sections()`.

## How a section is found

Every `text` must occur verbatim in the rendered context. Templates may trim whitespace, so declare text
without leading or trailing whitespace. Vitals locates each section with the most precise anchor given:

1. `msg_index`: the section lives in the i-th message of the request (the system message is 0), and only that
   message is searched. If the message has not appeared yet at a turn, the section is skipped for that turn.
   Use it for text in observations and injected feedback, where the same words can recur elsewhere.
2. `within` and `occurrence`: the section is searched inside the n-th copy of a longer enclosing text. Use it
   for byte-identical text that recurs, such as a repeated reminder.
3. Otherwise the first occurrence in the context is used.

A section that cannot be found is left out of that turn's read map. It is not an error, so check a declaration
before relying on it. Sections may overlap, and each is measured on its own tokens. The linear-layer
measures need a partition, so there the innermost section wins.

## Check a declaration

```bash
vitals annotate --check sections.json system_prompt.txt          # against a prompt file
vitals annotate --check sections.json runs/traces/ep-0001 \
  --gguf /path/to/model.gguf                                     # against a stored episode, rendered
```

Each section is reported as `ok`, `missing` (not found), `later` (anchored to a message the episode does not
reach), or `ambiguous` (found, but its text occurs more than once without an anchor, so the first copy is
used). The last line gives the share of the prompt the declaration covers. The command exits with status 1 if
any section is missing.

## Naming

- Name sections by function and keep names stable across harness versions, so read maps of different versions
  stay comparable: `task_framing`, `format_contract`, `directive:2`, `completion_check`.
- Group related sections with a prefix and a separator, such as `rules/refunds` and `rules/privacy`.
  `vitals replay --linear-group-sep /` pools the linear-layer measures by that prefix.
- Put the subject in the name when a harness repeats one kind of section per item, such as `cases:<label>`, so
  reading can be tied to the item.
- Prefer many small sections to a few large ones. A large section averages a hot line with the text around it,
  although peak lift recovers part of that.
