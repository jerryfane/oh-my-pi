/**
 * Wire definitions of the `topics_*` commands (owner-facing topics; see
 * `topics/host`), spread into the command table and the state definitions.
 * Refusals with a machine-readable reason carry `code` (`suggestion_not_found`,
 * `already_accepted`, `invalid_message_ids`, `organise_running`, `topics_disabled`,
 * `seed_too_large`).
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
	TopicSuggestion: doc(
		{
			id: doc("string", "Suggestion id (`sg_…`)."),
			title: "string",
			reason: "string",
			messageIds: doc("string[]", "Session entry ids of the Inbox messages the suggestion covers."),
			createdMs: "number.integer",
		},
		"An open topic suggestion of the Inbox model.",
	),
} satisfies WireDefs;

export const topicsRpcCommands: readonly RpcCommandSpec[] = [
	{
		name: "topics_list",
		doc: "List the owner's topics (persistent workers), newest activity first, and the open topic suggestions.",
		result: { topics: "TopicSummary[]", suggestions: "TopicSuggestion[]" },
	},
	{
		name: "topics_new",
		doc: "Start a topic with a title and its first message, optionally seeded with Inbox messages.",
		params: {
			title: "string",
			message: "string",
			"seedMessageIds?": doc("string[]", "Inbox user/assistant message entry ids copied into the topic."),
		},
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
	{
		name: "topics_accept_suggestion",
		doc: "Start the suggested topic, seeded with all of its Inbox messages; returns before the seed is ready.",
		params: { suggestionId: "string", by: doc("string", "Who accepted, as the host names them.") },
		result: { topicId: "string", alreadyAccepted: "boolean" },
	},
	{
		name: "topics_dismiss_suggestion",
		doc: "Dismiss a topic suggestion.",
		params: { suggestionId: "string", by: doc("string", "Who dismissed, as the host names them.") },
		result: { alreadyDismissed: "boolean" },
	},
	{
		name: "topics_organise",
		doc: "Queue one Inbox turn that reviews the history and suggests topics; never interrupts a running turn.",
		params: { by: doc("string", "Who asked, as the host names them.") },
		result: { queued: doc("boolean", "The Inbox was busy; the turn runs after the current one.") },
	},
];
