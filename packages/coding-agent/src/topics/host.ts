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
import type { SessionEntry } from "../session/session-entries";
import type { ToolSession } from "../tools";
import type { WorkerChange, WorkerSummary } from "../vibe/worker-scope";
import { type InboxBinding, InboxChangedError } from "./binding";
import { sessionInbox, type TopicsInbox } from "./inbox";
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

/** A topics operation admitted in one Inbox transcript: the Inbox, its binding and its topics' owner scope. */
interface Admission {
	box: TopicsInbox;
	binding: InboxBinding;
	scope: { ownerId: string; parentSessionId: string };
}

/** An operation on a suggestion while it runs, and the transcript it runs in. */
interface InFlight<T> {
	bindingKey: string;
	promise: Promise<T>;
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
	/** Announce a suggestion change to the owner scope of the Inbox transcript it happened in. */
	const emitSuggestion = (id: string, reason: SuggestionChange["reason"], { scope }: Admission): void => {
		for (const listener of suggestionListeners) listener({ id, reason, ...scope });
	};
	/** Suggestions made but not recorded yet (the Inbox turn that made them is still running), by transcript. */
	const unsaved: Array<{ bindingKey: string; suggestion: TopicSuggestion }> = [];
	/** The whole accept of a suggestion (topic, accepted state, note) while it runs, by suggestion id. */
	const accepting = new Map<string, InFlight<string>>();
	/**
	 * Topics started for accepts whose accepted state was not saved, by suggestion id: a retry in the same
	 * transcript saves the state for that topic instead of starting another, while that Inbox still lists it.
	 */
	const startedFor = new Map<string, { bindingKey: string; topicId: string }>();
	/** Dismissals being saved, by suggestion id. */
	const dismissing = new Map<string, InFlight<void>>();

	/** Bind an operation to the open Inbox transcript and the owner scope its topics live in. */
	const admit = (): Admission => {
		const box = requireInbox(inbox);
		return { box, binding: box.bind(), scope: registry.ownerScope(session) };
	};
	const notInThisInbox = (suggestionId: string): TopicsError =>
		new TopicsError("suggestion_not_found", `Suggestion ${suggestionId} belongs to an Inbox session that is no longer open.`);
	/** Whether the open Inbox's topics list `topicId`. */
	const listed = (topicId: string): boolean => registry.describe(session).some(topic => topic.id === topicId);
	/** The in-flight operation on `suggestionId`, if it runs in this transcript; one of another transcript refuses. */
	const inFlight = <T>(map: Map<string, InFlight<T>>, { binding }: Admission, suggestionId: string) => {
		const running = map.get(suggestionId);
		if (running && running.bindingKey !== binding.key) throw notInThisInbox(suggestionId);
		return running?.promise;
	};

	/** The suggestions of the bound transcript: its journal, plus those still being recorded there. */
	const folded = ({ box, binding }: Admission) =>
		foldSuggestions(
			box.entries(binding),
			unsaved.filter(entry => entry.bindingKey === binding.key).map(entry => entry.suggestion),
		);
	/** Whether `suggestionId` is being accepted in the bound transcript. */
	const acceptingHere = ({ binding }: Admission, suggestionId: string): boolean =>
		accepting.get(suggestionId)?.bindingKey === binding.key;

	/**
	 * Save a state change in the bound transcript (all or nothing; `check` runs under the write lock too) and
	 * announce it to that transcript's owner scope. Another open transcript refuses it: the suggestion is not
	 * part of that conversation.
	 */
	const commitState = async (admission: Admission, data: TopicSuggestionStateData, check?: () => void) => {
		try {
			await admission.box.appendState(admission.binding, TOPIC_SUGGESTION_STATE_TYPE, data, check);
		} catch (error) {
			if (error instanceof InboxChangedError) throw notInThisInbox(data.id);
			throw error;
		}
		emitSuggestion(data.id, data.state === "accepted" ? "suggestion-accepted" : "suggestion-dismissed", admission);
	};

	/**
	 * Spawn a topic; with `seed`, the chosen messages of `seed.entries` are prepended to its first turn. The
	 * seed is built lazily inside that turn, so its failure fails the turn and is never left unhandled.
	 */
	const spawnTopic = async (args: {
		title: string;
		message: string;
		name?: string;
		seed?: { entries: readonly SessionEntry[]; messageIds: string[]; reason?: string };
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
						seed: () =>
							buildTopicSeed({
								entries: seed.entries,
								messageIds: seed.messageIds,
								reason: seed.reason,
								summarize,
							}),
					}
				: {}),
		});
		return id;
	};

	/** The visible "Continued in topic" note, in the bound transcript only. */
	const noteContinued = ({ box, binding }: Admission, details: TopicContinuedDetails): void => {
		void box.appendNote(binding, {
			customType: TOPIC_CONTINUED_TYPE,
			content: `Continued in topic ${details.title}`,
			details,
		});
	};

	/**
	 * Accept `suggestion` of the bound transcript: start its topic (once per transcript, even across retries),
	 * save the accepted state, announce it and note it. Every read and write goes to that transcript; another
	 * open transcript, or a topic its Inbox no longer lists, refuses the accept.
	 */
	const acceptOnce = async (admission: Admission, suggestion: TopicSuggestion, by: string): Promise<string> => {
		const { box, binding } = admission;
		const started = startedFor.get(suggestion.id);
		let topicId: string;
		if (started) {
			if (started.bindingKey !== binding.key || !listed(started.topicId)) {
				startedFor.delete(suggestion.id);
				throw notInThisInbox(suggestion.id);
			}
			topicId = started.topicId;
		} else {
			topicId = await spawnTopic({
				title: suggestion.title,
				message: prompt.render(topicAcceptedTemplate, { title: suggestion.title }).trim(),
				seed: { entries: [...box.entries(binding)], messageIds: suggestion.messageIds, reason: suggestion.reason },
			});
			startedFor.set(suggestion.id, { bindingKey: binding.key, topicId });
		}
		const linked = topicId;
		await commitState(admission, { id: suggestion.id, state: "accepted", topicId, by, atMs: Date.now() }, () => {
			// Never link a topic the bound Inbox does not list.
			if (listed(linked)) return;
			startedFor.delete(suggestion.id);
			throw notInThisInbox(suggestion.id);
		});
		startedFor.delete(suggestion.id);
		noteContinued(admission, {
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
			const admission = admit();
			const { box, binding } = admission;
			requireEligibleMessageIds(seedMessageIds, box.branch(binding));
			const id = await spawnTopic({
				title,
				message,
				name,
				seed: { entries: [...box.entries(binding)], messageIds: seedMessageIds },
			});
			noteContinued(admission, {
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
			const admission = admit();
			return [...folded(admission).values()]
				.filter(suggestion => suggestion.state === "open" && !acceptingHere(admission, suggestion.id))
				.map(suggestionSummary);
		},
		suggestionBoard() {
			if (!inbox) return { open: [], dismissed: [] };
			const admission = admit();
			const all = [...folded(admission).values()];
			return {
				open: all
					.filter(suggestion => suggestion.state === "open" && !acceptingHere(admission, suggestion.id))
					.map(suggestionSummary),
				dismissed: all
					.filter(suggestion => suggestion.state === "dismissed")
					.sort((a, b) => (b.stateOrder ?? 0) - (a.stateOrder ?? 0))
					.slice(0, BOARD_DISMISSED_TITLES)
					.map(suggestion => suggestion.title),
			};
		},
		suggest({ title, reason, messageIds }) {
			const admission = admit();
			const { box, binding } = admission;
			const trimmedTitle = title.trim();
			const trimmedReason = reason.trim();
			if (!trimmedTitle) throw new Error("Suggestion title must not be empty.");
			if (!trimmedReason) throw new Error("Suggestion reason must not be empty.");
			const branch = box.branch(binding);
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
			const pending = { bindingKey: binding.key, suggestion };
			unsaved.push(pending);
			void box
				.appendNote(binding, {
					customType: TOPIC_SUGGESTION_TYPE,
					content: `Suggested topic: ${suggestion.title}. ${suggestion.reason}`,
					details: suggestion,
				})
				.then(saved => {
					// From here the session journal alone says whether the suggestion exists; only a saved one is announced.
					unsaved.splice(unsaved.indexOf(pending), 1);
					if (saved) emitSuggestion(suggestion.id, "suggested", admission);
				});
			return suggestion;
		},
		async acceptSuggestion({ suggestionId, by }) {
			const admission = admit();
			const running = inFlight(accepting, admission, suggestionId);
			if (running) return { topicId: await running, alreadyAccepted: true };
			const suggestion = folded(admission).get(suggestionId);
			if (!suggestion) throw new TopicsError("suggestion_not_found", `Unknown suggestion: ${suggestionId}`);
			if (suggestion.state === "accepted" && suggestion.topicId) {
				return { topicId: suggestion.topicId, alreadyAccepted: true };
			}
			const operation = acceptOnce(admission, suggestion, by);
			accepting.set(suggestionId, { bindingKey: admission.binding.key, promise: operation });
			try {
				return { topicId: await operation, alreadyAccepted: false };
			} finally {
				accepting.delete(suggestionId);
			}
		},
		async dismissSuggestion({ suggestionId, by }) {
			const admission = admit();
			const running = inFlight(dismissing, admission, suggestionId);
			if (running) {
				await running;
				return { alreadyDismissed: true };
			}
			const suggestion = folded(admission).get(suggestionId);
			if (!suggestion) throw new TopicsError("suggestion_not_found", `Unknown suggestion: ${suggestionId}`);
			const started = startedFor.get(suggestionId);
			if (
				suggestion.state === "accepted" ||
				acceptingHere(admission, suggestionId) ||
				started?.bindingKey === admission.binding.key
			) {
				throw new TopicsError("already_accepted", `Suggestion ${suggestionId} was already accepted.`);
			}
			// The journal keeps only saved states: a dismissal whose write failed left nothing, so it is retried.
			if (suggestion.state === "dismissed") return { alreadyDismissed: true };
			const operation = commitState(admission, { id: suggestionId, state: "dismissed", by, atMs: Date.now() });
			dismissing.set(suggestionId, { bindingKey: admission.binding.key, promise: operation });
			try {
				await operation;
				return { alreadyDismissed: false };
			} finally {
				dismissing.delete(suggestionId);
			}
		},
		async organise({ by }) {
			const { box, binding } = admit();
			if (box.turnPending(TOPIC_ORGANISE_TYPE)) {
				throw new TopicsError("organise_running", "An Organise turn is already queued or running.");
			}
			// Rendered right before dispatch, from the bound transcript only.
			return box.queueTurn(binding, TOPIC_ORGANISE_TYPE, () => {
				const index = formatMessageIndex(box.branch(binding), ORGANISE_INDEX_MESSAGES);
				return {
					customType: TOPIC_ORGANISE_TYPE,
					content: prompt.render(topicOrganiseTemplate, { index: index || "(none)" }).trim(),
					details: { by },
				};
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
