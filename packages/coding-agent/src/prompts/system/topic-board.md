<topic-board>
{{#if inbox}}
Topics: the owner's persistent side conversations, each run by its own agent next to this session.
{{else}}
Other topics of your owner: persistent side conversations, each run by its own agent. You are a topic too; you are not listed.
{{/if}}
Open topics first, newest activity first, then closed ones{{#if overflow}}; past the first {{cap}}, titles only{{/if}}. Times are UTC.
{{rows}}
Read a topic's conversation: `read history://<id>`. Message a topic: write to `agent://<id>`. A closed topic is paused by the owner; NEVER message it.
</topic-board>
