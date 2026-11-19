<topic-board>
{{#if inbox}}
Topics: the owner's persistent side conversations, each run by its own agent next to this session.
{{else}}
Other topics of your owner: persistent side conversations, each run by its own agent. You are a topic too; you are not listed.
{{/if}}
Open topics first, newest activity first, then closed ones{{#if overflow}}; past the first {{cap}}, titles only{{/if}}. Times are UTC.
{{rows}}
Read a topic's conversation: `read history://<id>`. Ask a topic: write to `agent://<id>`; it answers in a turn of its own (after the one it is working on) and the answer comes back to you when that turn ends. A closed topic refuses messages.
</topic-board>
