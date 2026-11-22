Suggest to the owner that a subject of this conversation move into its own topic (a persistent side conversation run by its own agent). It creates nothing: the owner sees the suggestion and starts the topic or dismisses it.

Use it with restraint:
- only when a clearly separate subject has started that will run for more than a couple of turns;
- at most one open suggestion per subject (the topic board lists the open ones and those dismissed: do not suggest a dismissed subject again unless the owner asks);
- never for one-off questions, or for what an existing topic already covers.

`message_ids` are the session entry ids of the user and assistant messages of this conversation that belong to the subject. Your pick is exactly what gets copied into the topic when the owner starts it (the owner cannot add or remove messages), so name every message related to the subject, and only those. If you do not know the ids, call with `message_ids: []`: the call fails and lists the latest messages with their ids.
