/**
 * Topic suggestions in the Inbox's session file. Session files are append-only,
 * so a suggestion is one visible `topic-suggestion` custom message and each
 * state change a hidden `topic-suggestion-state` custom entry; the latest state
 * entry of an id wins ({@link foldSuggestions}). Accepting one (or seeding a
 * topic by hand) adds a visible `topic-continued` note.
 */
import * as crypto from "node:crypto";
import { escapeXmlText } from "@oh-my-pi/pi-utils";
import { oneLineLabel } from "@oh-my-pi/pi-tui/tools/task";
import { escapeHarnessTags } from "../session/harness-tags";
import type { CustomEntry, CustomMessageEntry, SessionEntry } from "../session/session-entries";

/** `customType` of the visible suggestion message (`details` = {@link TopicSuggestionDetails}). */
export const TOPIC_SUGGESTION_TYPE = "topic-suggestion";
/** `customType` of the hidden state entry (`data` = {@link TopicSuggestionStateData}). */
export const TOPIC_SUGGESTION_STATE_TYPE = "topic-suggestion-state";
/** `customType` of the visible "Continued in topic" note (`details` = {@link TopicContinuedDetails}). */
export const TOPIC_CONTINUED_TYPE = "topic-continued";
/** `customType` of the hidden Organise instruction that starts one Inbox turn. */
export const TOPIC_ORGANISE_TYPE = "topic-organise";

/** Machine-readable failure reasons, sent as `code` next to `error`. */
export type TopicsErrorCode =
	| "suggestion_not_found"
	| "already_accepted"
	| "invalid_message_ids"
	| "organise_running"
	| "topics_disabled";

/** A refusal with a {@link TopicsErrorCode}; the message stays the human-readable `error`. */
export class TopicsError extends Error {
	readonly code: TopicsErrorCode;
	constructor(code: TopicsErrorCode, message: string) {
		super(message);
		this.name = "TopicsError";
		this.code = code;
	}
}

/** The code of a {@link TopicsError}, else undefined. */
export function topicsErrorCode(error: unknown): TopicsErrorCode | undefined {
	return error instanceof TopicsError ? error.code : undefined;
}

/** `details` of a {@link TOPIC_SUGGESTION_TYPE} message; also a suggestion as `topics_list` reports it. */
export interface TopicSuggestion {
	id: string;
	title: string;
	reason: string;
	messageIds: string[];
	createdMs: number;
}

export type TopicSuggestionDetails = TopicSuggestion;

/** `data` of a {@link TOPIC_SUGGESTION_STATE_TYPE} entry. */
export interface TopicSuggestionStateData {
	id: string;
	state: "accepted" | "dismissed";
	topicId?: string;
	by: string;
	atMs: number;
}

/** `details` of a {@link TOPIC_CONTINUED_TYPE} note. */
export interface TopicContinuedDetails {
	topicId: string;
	title: string;
	suggestionId: string | null;
	messageIds: string[];
}

/** A suggestion with its latest state. */
export interface FoldedSuggestion extends TopicSuggestion {
	state: "open" | "accepted" | "dismissed";
	topicId?: string;
	by?: string;
	/** Position of the latest state entry among all state entries (higher = later); absent while open. */
	stateOrder?: number;
}

/** A fresh suggestion id. */
export function newSuggestionId(): string {
	return `sg_${crypto.randomBytes(9).toString("base64url")}`;
}

function isStringArray(value: unknown): value is string[] {
	return Array.isArray(value) && value.every(item => typeof item === "string");
}

/** The suggestion an entry records, when it is a well-formed {@link TOPIC_SUGGESTION_TYPE} message. */
export function suggestionOf(entry: SessionEntry): TopicSuggestion | undefined {
	if (entry.type !== "custom_message" || entry.customType !== TOPIC_SUGGESTION_TYPE) return undefined;
	const details = (entry as CustomMessageEntry<Partial<TopicSuggestion>>).details;
	if (
		!details ||
		typeof details.id !== "string" ||
		typeof details.title !== "string" ||
		typeof details.reason !== "string" ||
		!isStringArray(details.messageIds) ||
		typeof details.createdMs !== "number"
	) {
		return undefined;
	}
	return {
		id: details.id,
		title: details.title,
		reason: details.reason,
		messageIds: details.messageIds,
		createdMs: details.createdMs,
	};
}

function stateOf(entry: SessionEntry): TopicSuggestionStateData | undefined {
	if (entry.type !== "custom" || entry.customType !== TOPIC_SUGGESTION_STATE_TYPE) return undefined;
	const data = (entry as CustomEntry<Partial<TopicSuggestionStateData>>).data;
	if (!data || typeof data.id !== "string" || (data.state !== "accepted" && data.state !== "dismissed")) {
		return undefined;
	}
	return {
		id: data.id,
		state: data.state,
		...(typeof data.topicId === "string" ? { topicId: data.topicId } : {}),
		by: typeof data.by === "string" ? data.by : "",
		atMs: typeof data.atMs === "number" ? data.atMs : 0,
	};
}

/**
 * Every suggestion in `entries` (file order) with its latest state, in the
 * order the suggestions were made. `unsaved` are suggestions made but not yet
 * in the file; a saved one wins over its unsaved copy. A state entry may
 * precede its suggestion (accepted before the suggestion was saved).
 */
export function foldSuggestions(
	entries: readonly SessionEntry[],
	unsaved: readonly TopicSuggestion[] = [],
): Map<string, FoldedSuggestion> {
	const made = new Map<string, TopicSuggestion>();
	const states = new Map<string, { data: TopicSuggestionStateData; order: number }>();
	let order = 0;
	for (const entry of entries) {
		const suggestion = suggestionOf(entry);
		if (suggestion) {
			if (!made.has(suggestion.id)) made.set(suggestion.id, suggestion);
			continue;
		}
		const state = stateOf(entry);
		if (state) states.set(state.id, { data: state, order: order++ });
	}
	for (const suggestion of unsaved) if (!made.has(suggestion.id)) made.set(suggestion.id, suggestion);
	const folded = new Map<string, FoldedSuggestion>();
	for (const suggestion of [...made.values()].sort((a, b) => a.createdMs - b.createdMs)) {
		const latest = states.get(suggestion.id);
		folded.set(
			suggestion.id,
			latest
				? {
						...suggestion,
						state: latest.data.state,
						...(latest.data.topicId !== undefined ? { topicId: latest.data.topicId } : {}),
						by: latest.data.by,
						stateOrder: latest.order,
					}
				: { ...suggestion, state: "open" },
		);
	}
	return folded;
}

/** A suggestion as `topics_list` reports it. */
export function suggestionSummary(suggestion: TopicSuggestion): TopicSuggestion {
	const { id, title, reason, messageIds, createdMs } = suggestion;
	return { id, title, reason, messageIds: [...messageIds], createdMs };
}

/** The text of a user or assistant message entry (text blocks only), or undefined for any other entry. */
export function messageText(entry: SessionEntry): { role: "user" | "assistant"; text: string } | undefined {
	if (entry.type !== "message") return undefined;
	const message = entry.message;
	if (message.role !== "user" && message.role !== "assistant") return undefined;
	const content = message.content;
	const text =
		typeof content === "string"
			? content
			: content
					.map(block => (block.type === "text" ? block.text : ""))
					.filter(Boolean)
					.join("\n");
	return text.trim() ? { role: message.role, text } : undefined;
}

/**
 * The ids a suggestion or a seed may name: the user and assistant text messages of `branch` (the Inbox's
 * current branch, after its latest `/clear`). Build it once per branch read; listing, accept and create share it.
 */
export function eligibleMessageIds(branch: readonly SessionEntry[]): Set<string> {
	const ids = new Set<string>();
	for (const entry of branch) if (hasMessageText(entry)) ids.add(entry.id);
	return ids;
}

const NONBLANK = /\S/;

/** Whether {@link messageText} has text for `entry` (a user or assistant message with nonblank text), without building it. */
function hasMessageText(entry: SessionEntry): boolean {
	if (entry.type !== "message") return false;
	const message = entry.message;
	if (message.role !== "user" && message.role !== "assistant") return false;
	const content = message.content;
	if (typeof content === "string") return NONBLANK.test(content);
	return content.some(block => block.type === "text" && NONBLANK.test(block.text));
}

/** Check that every id in `ids` is in `eligible` ({@link eligibleMessageIds}); throws `invalid_message_ids` otherwise. */
export function requireEligibleMessageIds(ids: readonly string[], eligible: ReadonlySet<string>): void {
	const invalid = ids.filter(id => !eligible.has(id));
	if (invalid.length > 0) {
		throw new TopicsError(
			"invalid_message_ids",
			`Not user or assistant text messages of the Inbox's current conversation: ${invalid.join(", ")}.`,
		);
	}
}

/** Longest excerpt of a message in {@link formatMessageIndex}. */
const INDEX_EXCERPT_MAX = 100;

/**
 * The newest `limit` user and assistant text messages of `branch` as
 * `- <id> · <role> · "<start of text>"` lines, oldest first, so the model can
 * name them; empty when there are none.
 */
export function formatMessageIndex(branch: readonly SessionEntry[], limit: number): string {
	const rows: string[] = [];
	for (const entry of branch) {
		const message = messageText(entry);
		if (!message) continue;
		const excerpt = escapeHarnessTags(JSON.stringify(oneLineLabel(message.text, INDEX_EXCERPT_MAX)));
		rows.push(`- ${escapeXmlText(entry.id)} · ${message.role} · ${excerpt}`);
	}
	const omitted = rows.length - limit;
	const shown = omitted > 0 ? rows.slice(omitted) : rows;
	return omitted > 0 ? [`(${omitted} older messages not listed)`, ...shown].join("\n") : shown.join("\n");
}

/** Open suggestions and recently dismissed titles, as the Inbox's topic board lists them. */
export interface SuggestionBoard {
	open: TopicSuggestion[];
	/** Titles of the latest dismissed suggestions, newest first. */
	dismissed: string[];
}
