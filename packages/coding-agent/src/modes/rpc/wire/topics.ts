/**
 * Wire definitions of the `topics_*` commands (owner-facing topics; see
 * `topics/host`), spread into the command table and the state definitions.
 */
import type { RpcCommandSpec } from "./commands";
import { doc, type WireDefs } from "./dsl";

export const topicsStateDefs = {
	TopicSummary: {
		id: "string",
		title: "string",
		state: "'starting' | 'running' | 'idle' | 'dead' | 'closed'",
		turns: "number.integer",
		queued: "number.integer",
		createdAt: "number.integer",
		lastActivityAt: "number.integer",
		"lastActivity?": "string",
		"ownerLine?": doc("string", "First line of the owner's latest message to the topic, when this process saw one."),
		"sessionFile?": "string",
		"turnStartedAt?": doc("number.integer", "When the turn in flight started; absent while no turn runs."),
	},
} satisfies WireDefs;

export const topicsRpcCommands: readonly RpcCommandSpec[] = [
	{
		name: "topics_list",
		doc: "List the owner's topics (persistent workers), newest activity first.",
		result: { topics: "TopicSummary[]" },
		unwrap: "topics",
	},
	{
		name: "topics_new",
		doc: "Start a topic with a title and its first message.",
		params: { title: "string", message: "string" },
		result: { topicId: "string" },
		unwrap: "topicId",
	},
	{
		name: "topics_send",
		doc: "Send the owner's message to one topic: steers a running turn, queues behind one, or starts a new turn.",
		params: { topicId: "string", message: "string" },
		result: { topicId: "string", mode: "'steered' | 'queued' | 'turn'" },
	},
	{
		name: "topics_read",
		doc: "Read a topic's own conversation incrementally.",
		params: { topicId: "string", "fromByte?": "number.integer" },
		result: "SubagentMessages",
	},
	{
		name: "topics_close",
		doc: "Close a topic; its conversation is kept and it can be reopened.",
		params: { topicId: "string" },
	},
	{
		name: "topics_reopen",
		doc: "Reopen a closed topic.",
		params: { topicId: "string" },
	},
	{
		name: "topics_rename",
		doc: "Change a topic's title.",
		params: { topicId: "string", title: "string" },
	},
];
