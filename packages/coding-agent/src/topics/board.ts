/**
 * Topic board: the owner's topics as one hidden context message. A session
 * gets a fresh board ahead of a turn only when its text differs from the last
 * board in its history, so an unchanged board adds nothing and a change appends
 * one message at the end: the cached request prefix never moves, and the board
 * is persisted at the slot it was sent from, so a resumed process sends it there too.
 */
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import { escapeXmlText, prompt } from "@oh-my-pi/pi-utils";
import { oneLineLabel } from "@oh-my-pi/pi-tui/tools/task";
import topicBoardPrompt from "../prompts/system/topic-board.md" with { type: "text" };
import type { CustomMessage } from "../session/messages";
import type { WorkerSummary } from "../vibe/runtime";

/** `customType` of the topic board message. */
export const TOPIC_BOARD_CONTEXT_TYPE = "topic-board";

/** Topics listed in full; the rest are listed by title only. */
export const TOPIC_BOARD_FULL_ROWS = 40;

const FIELD_MAX = 120;

/** Whether `message` is a topic board. */
export function isTopicBoardContext(message: AgentMessage | undefined): message is CustomMessage {
	return message?.role === "custom" && message.customType === TOPIC_BOARD_CONTEXT_TYPE;
}

/** Who reads the board: the Inbox (main session), or a topic seeing its siblings. */
export type TopicBoardAudience = "inbox" | "topic";

function field(text: string): string {
	return escapeXmlText(oneLineLabel(text, FIELD_MAX));
}

function stateLabel(topic: WorkerSummary): string {
	if (topic.state === "closed") return "closed";
	const base = topic.state === "idle" ? "idle" : "working";
	return topic.queued > 0 ? `${base}, queued ${topic.queued}` : base;
}

/** Minute-precision UTC, so the board text only moves when a topic does. */
function timeLabel(ms: number): string {
	return Number.isFinite(ms) ? `${new Date(ms).toISOString().slice(0, 16)}Z` : "unknown";
}

function byNewestActivity(a: WorkerSummary, b: WorkerSummary): number {
	return b.lastActivityAt - a.lastActivityAt || a.id.localeCompare(b.id);
}

/**
 * The board text for `topics`, or `undefined` when there are none. One row per
 * topic: open ones first, then closed ones, each newest activity first; past
 * {@link TOPIC_BOARD_FULL_ROWS} rows, title and id only.
 */
export function formatTopicBoard(topics: readonly WorkerSummary[], audience: TopicBoardAudience): string | undefined {
	if (topics.length === 0) return undefined;
	const open = topics.filter(topic => topic.state !== "closed").sort(byNewestActivity);
	const closed = topics.filter(topic => topic.state === "closed").sort(byNewestActivity);
	const rows = [...open, ...closed].map((topic, index) => {
		const head = `- ${field(topic.title)} (id: ${field(topic.id)})`;
		if (index >= TOPIC_BOARD_FULL_ROWS) return head;
		const gist = topic.lastActivity ? `; ${field(topic.lastActivity)}` : "";
		return `${head}: ${stateLabel(topic)}; last activity ${timeLabel(topic.lastActivityAt)}${gist}`;
	});
	return prompt
		.render(topicBoardPrompt, {
			inbox: audience === "inbox",
			overflow: rows.length > TOPIC_BOARD_FULL_ROWS,
			cap: TOPIC_BOARD_FULL_ROWS,
			rows: rows.join("\n"),
		})
		.trim();
}

/**
 * The board message to send ahead of the next turn of a session whose history
 * is `history`, or `undefined` when there is no board or the latest board in
 * `history` already says exactly this.
 */
export function topicBoardMessage(
	board: string | undefined,
	history: readonly AgentMessage[],
): CustomMessage | undefined {
	if (!board || history.findLast(isTopicBoardContext)?.content === board) return undefined;
	return {
		role: "custom",
		customType: TOPIC_BOARD_CONTEXT_TYPE,
		content: board,
		display: false,
		attribution: "agent",
		timestamp: Date.now(),
	};
}
