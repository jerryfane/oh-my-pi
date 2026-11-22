/**
 * Host-facing topics surface: what a host (RPC client, Herdr control socket)
 * may do with the owner's topics. A topic is a persistent worker in the
 * {@link topicsRegistry} scope of the top-level session; the host talks to it
 * directly, without a turn of the main conversation. While a host lives,
 * agents' `agent://<id>` messages to its topics run as topic turns
 * ({@link routeTopicMessage}).
 *
 * The Inbox (the top-level session) also holds topic suggestions
 * (`topics/suggestions`): the Inbox model suggests, the owner accepts (a topic
 * seeded with the chosen Inbox messages) or dismisses, and Organise asks the
 * Inbox model for several suggestions in one turn.
 */
import { prompt } from "@oh-my-pi/pi-utils";
import { setIrcRoute } from "../irc/routes";
import { readRpcSubagentTranscript } from "../modes/rpc/rpc-subagents";
import type { RpcSubagentMessagesResult } from "../modes/rpc/rpc-types";
import topicAcceptedTemplate from "../prompts/system/topic-accepted.md" with { type: "text" };
import topicOrganiseTemplate from "../prompts/system/topic-organise.md" with { type: "text" };
import type { AgentSession } from "../session/agent-session";
import type { ToolSession } from "../tools";
import type { WorkerChange, WorkerSummary } from "../vibe/worker-scope";
import { InboxChangedError, sessionInbox, type TopicsInbox } from "./inbox";
import { buildTopicSeed, type SeedSummarizer, smolSeedSummarizer } from "./seed";
import { cfgTopicsEnabled } from "./settings";
import {
	foldSuggestions,
	formatMessageIndex,
	newSuggestionId,
	requireEligibleMessageIds,
	type SuggestionBoard,
	suggestionSummary,
	TOPIC_CONTINUED_TYPE,
	TOPIC_ORGANISE_TYPE,
	TOPIC_SUGGESTION_STATE_TYPE,
	TOPIC_SUGGESTION_TYPE,
	type TopicContinuedDetails,
	TopicsError,
	type TopicSuggestion,
	type TopicSuggestionStateData,
} from "./suggestions";
import { routeTopicMessage, topicsRegistry } from "./workers";

/** One topic as a host sees it. */
export type TopicSummary = WorkerSummary;

/** Outcome of sending to a topic: steered into a running turn, queued behind one, or a new turn. */
export interface TopicSendOutcome {
	id: string;
	mode: "steered" | "queued" | "turn";
}

/** A suggestion was made (saved in the Inbox), accepted or dismissed; `id` is the suggestion's. */
export interface SuggestionChange {
	id: string;
	ownerId: string;
	parentSessionId: string;
	reason: "suggested" | "suggestion-accepted" | "suggestion-dismissed";
}

/** What `subscribe` streams to a host: a topic's change, or a suggestion's. */
export type TopicChange = WorkerChange | SuggestionChange;

/** Dismissed suggestion titles the Inbox board lists. */
export const BOARD_DISMISSED_TITLES = 10;
/** Inbox messages listed (with their ids) in the Organise instruction. */
const ORGANISE_INDEX_MESSAGES = 150;
/** Inbox messages listed when `topic_suggest` names none or wrong ones. */
const SUGGEST_INDEX_MESSAGES = 40;

export interface TopicsHost {
	list(): TopicSummary[];
	/** Start a topic with an owner-facing title and its first message, seeded with Inbox messages when given. */
	create(args: { title: string; message: string; name?: string; seedMessageIds?: string[] }): Promise<{ id: string }>;
	send(id: string, message: string): Promise<TopicSendOutcome>;
	/** Read the topic's own conversation from `fromByte` (incremental). */
	read(id: string, fromByte?: number): Promise<RpcSubagentMessagesResult>;
	close(id: string): Promise<void>;
	reopen(id: string): Promise<void>;
	rename(id: string, title: string): Promise<void>;
	/** Open suggestions, oldest first. */
	suggestions(): TopicSuggestion[];
	/** Open suggestions and the latest dismissed titles, for the Inbox's topic board. */
	suggestionBoard(): SuggestionBoard;
	/** Record the Inbox model's suggestion (`topic_suggest`); creates nothing. */
	suggest(args: { title: string; reason: string; messageIds: string[] }): TopicSuggestion;
	/** Start the suggested topic, seeded with all of the suggestion's messages; returns before the seed is ready. */
	acceptSuggestion(args: { suggestionId: string; by: string }): Promise<{ topicId: string; alreadyAccepted: boolean }>;
	dismissSuggestion(args: { suggestionId: string; by: string }): Promise<{ alreadyDismissed: boolean }>;
	/** Queue one Inbox turn that reviews the history and suggests topics. */
	organise(args: { by: string }): Promise<{ queued: boolean }>;
	/** Restore persisted topics of the current session (startup, session switch). */
	restore(): Promise<number>;
	onChange(listener: (change: TopicChange) => void): () => void;
	/** Stop taking agents' messages to the topics; the topics themselves are kept. */
	dispose(): void;
}

export interface TopicsHostOptions {
	/** The Inbox's session; suggestions, seeds and Organise need it. */
	inbox?: TopicsInbox;
	/** Condenses older seed messages; default one call to the `smol` role model. */
	summarize?: SeedSummarizer;
}

function requireInbox(inbox: TopicsInbox | undefined): TopicsInbox {
	if (!inbox) throw new Error("Topic suggestions need the Inbox session.");
	return inbox;
}

/**
 * Bind the topics registry to the top-level session's tool session. A second
 * host for the same session replaces the first one's message route.
 */
export function createTopicsHost(session: ToolSession, options: TopicsHostOptions = {}): TopicsHost {
	const { inbox } = options;
	const summarize = options.summarize ?? smolSeedSummarizer(session);
	const registry = topicsRegistry();
	const ownedBySession = (change: WorkerChange): boolean => {
		const scope = registry.ownerScope(session);
		return change.ownerId === scope.ownerId && change.parentSessionId === scope.parentSessionId;
	};
	const unroute = setIrcRoute(session, (message, bus) => routeTopicMessage(registry, session, message, bus));
	const suggestionListeners = new Set<(change: TopicChange) => void>();
	/** Announce a suggestion change to the owner scope `scope` of the Inbox session it happened in. */
	const emitSuggestion = (
		id: string,
		reason: SuggestionChange["reason"],
		{ ownerId, parentSessionId }: { ownerId: string; parentSessionId: string },
	): void => {
		for (const listener of suggestionListeners) listener({ id, ownerId, parentSessionId, reason });
	};
	/** Suggestions made but not recorded yet (the Inbox turn that made them is still running). */
	const unsaved: Array<{ sessionId: string; suggestion: TopicSuggestion }> = [];
	/** The whole accept of a suggestion (topic, accepted state, note) while it runs, by suggestion id. */
	const accepting = new Map<string, Promise<string>>();
	/** Topics started for suggestions whose accepted state is not saved yet: a retry saves it, never respawns. */
	const startedFor = new Map<string, string>();
	/** Dismissals being saved, by suggestion id. */
	const dismissing = new Map<string, Promise<void>>();

	/** The suggestions of the open Inbox session: its journal, plus those still being recorded. */
	const folded = (box: TopicsInbox) => {
		const sessionId = box.sessionId();
		return foldSuggestions(
			box.entries(),
			unsaved.filter(entry => entry.sessionId === sessionId).map(entry => entry.suggestion),
		);
	};

	/**
	 * Save a state change in Inbox session `sessionId` (all or nothing) and
	 * announce it to `scope`. Another open session refuses it: the suggestion is
	 * not part of that conversation.
	 */
	const commitState = async (
		box: TopicsInbox,
		sessionId: string,
		scope: { ownerId: string; parentSessionId: string },
		data: TopicSuggestionStateData,
	): Promise<void> => {
		try {
			await box.appendState(sessionId, TOPIC_SUGGESTION_STATE_TYPE, data);
		} catch (error) {
			if (!(error instanceof InboxChangedError)) throw error;
			throw new TopicsError(
				"suggestion_not_found",
				`Suggestion ${data.id} belongs to an Inbox session that is no longer open.`,
			);
		}
		emitSuggestion(data.id, data.state === "accepted" ? "suggestion-accepted" : "suggestion-dismissed", scope);
	};

	/** Spawn a topic; with `seed`, the chosen Inbox messages are prepended to its first turn once ready. */
	const spawnTopic = async (args: {
		title: string;
		message: string;
		name?: string;
		seed?: { box: TopicsInbox; messageIds: string[]; reason?: string };
	}): Promise<string> => {
		const title = args.title.trim();
		if (!title) throw new Error("Topic title must not be empty.");
		const seed = args.seed;
		const { id } = await registry.spawn(session, {
			cli: "good",
			name: args.name ?? title,
			prompt: args.message,
			title,
			...(seed
				? {
						seed: buildTopicSeed({
							entries: [...seed.box.entries()],
							messageIds: seed.messageIds,
							reason: seed.reason,
							summarize,
						}),
					}
				: {}),
		});
		return id;
	};

	/** The visible "Continued in topic" note, in Inbox session `sessionId` only. */
	const noteContinued = (box: TopicsInbox, sessionId: string, details: TopicContinuedDetails): void => {
		void box.appendNote(sessionId, {
			customType: TOPIC_CONTINUED_TYPE,
			content: `Continued in topic ${details.title}`,
			details,
		});
	};

	/**
	 * Accept `suggestion` of Inbox session `sessionId`: start its topic (once,
	 * even across retries), save the accepted state, announce it and note it.
	 * Every write goes to that session; another open session refuses the accept.
	 */
	const acceptOnce = async (
		box: TopicsInbox,
		sessionId: string,
		suggestion: TopicSuggestion,
		by: string,
	): Promise<string> => {
		const scope = registry.ownerScope(session);
		let topicId = startedFor.get(suggestion.id);
		if (!topicId) {
			topicId = await spawnTopic({
				title: suggestion.title,
				message: prompt.render(topicAcceptedTemplate, { title: suggestion.title }).trim(),
				seed: { box, messageIds: suggestion.messageIds, reason: suggestion.reason },
			});
			startedFor.set(suggestion.id, topicId);
		}
		await commitState(box, sessionId, scope, {
			id: suggestion.id,
			state: "accepted",
			topicId,
			by,
			atMs: Date.now(),
		});
		startedFor.delete(suggestion.id);
		noteContinued(box, sessionId, {
			topicId,
			title: suggestion.title,
			suggestionId: suggestion.id,
			messageIds: [...suggestion.messageIds],
		});
		return topicId;
	};

	return {
		list: () => registry.describe(session),
		async create({ title, message, name, seedMessageIds }) {
			if (!seedMessageIds?.length) return { id: await spawnTopic({ title, message, name }) };
			const box = requireInbox(inbox);
			const sessionId = box.sessionId();
			requireEligibleMessageIds(seedMessageIds, box.branch());
			const id = await spawnTopic({ title, message, name, seed: { box, messageIds: seedMessageIds } });
			noteContinued(box, sessionId, {
				topicId: id,
				title: title.trim(),
				suggestionId: null,
				messageIds: [...seedMessageIds],
			});
			return { id };
		},
		async send(id, message) {
			const outcome = await registry.send(session, { session: id, message });
			return { id: outcome.id, mode: outcome.mode };
		},
		async read(id, fromByte) {
			const topic = registry.describe(session).find(entry => entry.id === id);
			if (!topic?.sessionFile) throw new Error(`Unknown topic: ${id}`);
			return readRpcSubagentTranscript(topic.sessionFile, fromByte);
		},
		close: id => registry.close(session, id),
		reopen: id => registry.reopen(session, id),
		rename: (id, title) => registry.rename(session, id, title),
		suggestions() {
			if (!inbox) return [];
			return [...folded(inbox).values()]
				.filter(suggestion => suggestion.state === "open" && !accepting.has(suggestion.id))
				.map(suggestionSummary);
		},
		suggestionBoard() {
			if (!inbox) return { open: [], dismissed: [] };
			const all = [...folded(inbox).values()];
			return {
				open: all
					.filter(suggestion => suggestion.state === "open" && !accepting.has(suggestion.id))
					.map(suggestionSummary),
				dismissed: all
					.filter(suggestion => suggestion.state === "dismissed")
					.sort((a, b) => (b.stateOrder ?? 0) - (a.stateOrder ?? 0))
					.slice(0, BOARD_DISMISSED_TITLES)
					.map(suggestion => suggestion.title),
			};
		},
		suggest({ title, reason, messageIds }) {
			const box = requireInbox(inbox);
			const trimmedTitle = title.trim();
			const trimmedReason = reason.trim();
			if (!trimmedTitle) throw new Error("Suggestion title must not be empty.");
			if (!trimmedReason) throw new Error("Suggestion reason must not be empty.");
			const branch = box.branch();
			const index = formatMessageIndex(branch, SUGGEST_INDEX_MESSAGES);
			if (messageIds.length === 0) {
				throw new TopicsError(
					"invalid_message_ids",
					`Name the messages that belong to the topic (\`message_ids\`). Latest messages of this conversation:\n${index}`,
				);
			}
			try {
				requireEligibleMessageIds(messageIds, branch);
			} catch (error) {
				if (!(error instanceof TopicsError)) throw error;
				throw new TopicsError(error.code, `${error.message} Latest messages of this conversation:\n${index}`);
			}
			const suggestion: TopicSuggestion = {
				id: newSuggestionId(),
				title: trimmedTitle,
				reason: trimmedReason,
				messageIds: [...new Set(messageIds)],
				createdMs: Date.now(),
			};
			const sessionId = box.sessionId();
			const scope = registry.ownerScope(session);
			const pending = { sessionId, suggestion };
			unsaved.push(pending);
			void box
				.appendNote(sessionId, {
					customType: TOPIC_SUGGESTION_TYPE,
					content: `Suggested topic: ${suggestion.title}. ${suggestion.reason}`,
					details: suggestion,
				})
				.then(saved => {
					// From here the session journal alone says whether the suggestion exists; only a saved one is announced.
					unsaved.splice(unsaved.indexOf(pending), 1);
					if (saved) emitSuggestion(suggestion.id, "suggested", scope);
				});
			return suggestion;
		},
		async acceptSuggestion({ suggestionId, by }) {
			const box = requireInbox(inbox);
			const inFlight = accepting.get(suggestionId);
			if (inFlight) return { topicId: await inFlight, alreadyAccepted: true };
			const sessionId = box.sessionId();
			const suggestion = folded(box).get(suggestionId);
			if (!suggestion) throw new TopicsError("suggestion_not_found", `Unknown suggestion: ${suggestionId}`);
			if (suggestion.state === "accepted" && suggestion.topicId) {
				return { topicId: suggestion.topicId, alreadyAccepted: true };
			}
			const operation = acceptOnce(box, sessionId, suggestion, by);
			accepting.set(suggestionId, operation);
			try {
				return { topicId: await operation, alreadyAccepted: false };
			} finally {
				accepting.delete(suggestionId);
			}
		},
		async dismissSuggestion({ suggestionId, by }) {
			const box = requireInbox(inbox);
			const inFlight = dismissing.get(suggestionId);
			if (inFlight) {
				await inFlight;
				return { alreadyDismissed: true };
			}
			const sessionId = box.sessionId();
			const suggestion = folded(box).get(suggestionId);
			if (!suggestion) throw new TopicsError("suggestion_not_found", `Unknown suggestion: ${suggestionId}`);
			if (suggestion.state === "accepted" || accepting.has(suggestionId) || startedFor.has(suggestionId)) {
				throw new TopicsError("already_accepted", `Suggestion ${suggestionId} was already accepted.`);
			}
			// The journal keeps only saved states: a dismissal whose write failed left nothing, so it is retried.
			if (suggestion.state === "dismissed") return { alreadyDismissed: true };
			const operation = commitState(box, sessionId, registry.ownerScope(session), {
				id: suggestionId,
				state: "dismissed",
				by,
				atMs: Date.now(),
			});
			dismissing.set(suggestionId, operation);
			try {
				await operation;
				return { alreadyDismissed: false };
			} finally {
				dismissing.delete(suggestionId);
			}
		},
		async organise({ by }) {
			const box = requireInbox(inbox);
			if (box.turnPending(TOPIC_ORGANISE_TYPE)) {
				throw new TopicsError("organise_running", "An Organise turn is already queued or running.");
			}
			const index = formatMessageIndex(box.branch(), ORGANISE_INDEX_MESSAGES);
			return box.queueTurn({
				customType: TOPIC_ORGANISE_TYPE,
				content: prompt.render(topicOrganiseTemplate, { index: index || "(none)" }).trim(),
				details: { by },
			});
		},
		restore: () => registry.rehydrate(session),
		onChange(listener) {
			suggestionListeners.add(listener);
			const unsubscribe = registry.onChange(change => {
				if (ownedBySession(change)) listener(change);
			});
			return () => {
				suggestionListeners.delete(listener);
				unsubscribe();
			};
		},
		dispose: unroute,
	};
}

/**
 * The session's topics host factory: only for a top-level session with
 * `topics.enabled`. `inboxSession` returns the Inbox once it exists (the
 * factory runs on first use, after the session is built).
 */
export function topicsHostFactory(
	session: ToolSession,
	topLevel: boolean,
	inboxSession?: () => AgentSession,
): (() => TopicsHost) | undefined {
	if (!topLevel || !cfgTopicsEnabled.get(session.settings)) return undefined;
	return () => createTopicsHost(session, { inbox: inboxSession && sessionInbox(inboxSession()) });
}

/**
 * A top-level session's topics host: created on first use and disposed with
 * the session, its persisted topics restored once per parent session file,
 * so a session switch restores the new session's topics.
 */
export class SessionTopics {
	readonly #create: () => TopicsHost;
	readonly #addDisposer: (dispose: () => void) => void;
	#host: TopicsHost | undefined;
	/** Parent session file whose persisted topics were last restored (`undefined` = never). */
	#restoredFor: string | null | undefined;

	constructor(create: () => TopicsHost, addDisposer: (dispose: () => void) => void) {
		this.#create = create;
		this.#addDisposer = addDisposer;
	}

	/** The host, with the topics of `sessionFile` restored. */
	async host(sessionFile: string | null): Promise<TopicsHost> {
		if (!this.#host) {
			const host = this.#create();
			this.#host = host;
			this.#addDisposer(() => host.dispose());
		}
		if (this.#restoredFor !== sessionFile) {
			await this.#host.restore();
			this.#restoredFor = sessionFile;
		}
		return this.#host;
	}
}
