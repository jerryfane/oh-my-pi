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
import { logger, prompt } from "@oh-my-pi/pi-utils";
import { setIrcRoute } from "../irc/routes";
import { readRpcSubagentTranscript } from "../modes/rpc/rpc-subagents";
import type { RpcSubagentMessagesResult } from "../modes/rpc/rpc-types";
import topicAcceptedTemplate from "../prompts/system/topic-accepted.md" with { type: "text" };
import topicOrganiseTemplate from "../prompts/system/topic-organise.md" with { type: "text" };
import type { AgentSession } from "../session/agent-session";
import type { SessionEntry } from "../session/session-entries";
import type { ToolSession } from "../tools";
import type { WorkerChange, WorkerSummary } from "../vibe/worker-scope";
import { type InboxEpoch, StaleTranscriptError, throwIfStale } from "../session/inbox-epoch";
import { sessionInbox, type TopicsInbox } from "./inbox";
import { buildTopicSeed, type SeedSummarizer, smolSeedSummarizer } from "./seed";
import { cfgTopicsEnabled } from "./settings";
import {
	type FoldedSuggestion,
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

/** A topics operation admitted in one Inbox transcript: the Inbox, that transcript's epoch, and its topics' owner scope. */
interface Admission {
	box: TopicsInbox;
	epoch: InboxEpoch;
	scope: { ownerId: string; parentSessionId: string };
}

/** An operation on a suggestion while it runs, and the epoch (transcript) it runs in. */
interface InFlight<T> {
	epochId: number;
	promise: Promise<T>;
}

/** A spawned topic, and when its first turn was admitted (it rejects when that turn was cancelled or failed). */
interface StartedTopic {
	id: string;
	admitted: Promise<void>;
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
	/** Suggestions made but not recorded yet (the Inbox turn that made them is still running), by epoch. */
	const unsaved: Array<{ epochId: number; suggestion: TopicSuggestion }> = [];
	/**
	 * Accepts in progress, by suggestion id: the topic is spawned, the accepted state and the note wait for its
	 * first turn to be admitted. Until then the suggestion is pending in memory only.
	 */
	const accepting = new Map<string, InFlight<string>>();
	/** Dismissals being saved, by suggestion id. */
	const dismissing = new Map<string, InFlight<void>>();

	/** Bind a mutation to the open Inbox transcript; throws {@link StaleTranscriptError} while it is being replaced. */
	const admit = (): Admission => {
		const box = requireInbox(inbox);
		return { box, epoch: box.capture(), scope: registry.ownerScope(session) };
	};
	/** The open transcript's epoch id for a read, or undefined while it is being replaced. */
	const openEpochId = (box: TopicsInbox): number | undefined => {
		try {
			return box.capture().id;
		} catch {
			return undefined;
		}
	};
	const notInThisInbox = (suggestionId: string): TopicsError =>
		new TopicsError("suggestion_not_found", `Suggestion ${suggestionId} belongs to an Inbox session that is no longer open.`);
	/** The in-flight operation on `suggestionId` in this epoch; one of an aborted epoch is dead and ignored. */
	const inFlight = <T>(map: Map<string, InFlight<T>>, { epoch }: Admission, suggestionId: string) => {
		const running = map.get(suggestionId);
		return running?.epochId === epoch.id ? running.promise : undefined;
	};

	/** The suggestions of the open transcript: its journal, plus those still being recorded in epoch `epochId`. */
	const folded = (box: TopicsInbox, epochId: number | undefined) =>
		foldSuggestions(
			box.entries(),
			unsaved.filter(entry => entry.epochId === epochId).map(entry => entry.suggestion),
		);
	/** The message ids a seed may copy: the current branch after the latest `/clear` (one set per listing). */
	const eligibleIds = (box: TopicsInbox): Set<string> => new Set(box.branch().map(entry => entry.id));
	/** Whether `suggestion` is offered for acceptance: open, not being accepted here, and all its messages eligible. */
	const offered = (suggestion: FoldedSuggestion, eligible: ReadonlySet<string>, epochId: number | undefined): boolean =>
		suggestion.state === "open" &&
		accepting.get(suggestion.id)?.epochId !== epochId &&
		suggestion.messageIds.every(id => eligible.has(id));

	/** Refusals of a replaced transcript, as topics reports them for `suggestionId`. */
	const translateStale = (suggestionId: string) => (error: unknown) => {
		throw error instanceof StaleTranscriptError ? notInThisInbox(suggestionId) : error;
	};

	/**
	 * Save a state change in the admitted transcript (all or nothing) and announce it to that transcript's owner
	 * scope. A replaced transcript refuses it: the suggestion is not part of the one open now.
	 */
	const commitState = async (admission: Admission, data: TopicSuggestionStateData) => {
		await admission.box
			.appendState(admission.epoch, TOPIC_SUGGESTION_STATE_TYPE, data)
			.catch(translateStale(data.id));
		emitSuggestion(data.id, data.state === "accepted" ? "suggestion-accepted" : "suggestion-dismissed", admission);
	};

	/** End a topic whose first turn was cancelled or failed (best effort: only its own transcript can record it). */
	const endTopic = async (id: string, admission: Admission | undefined, turn: Promise<unknown> | undefined) => {
		await turn?.catch(() => {});
		await registry.kill(session, id).catch(error =>
			logger.debug("topics: could not end a cancelled topic", {
				id,
				error: error instanceof Error ? error.message : String(error),
				parentSessionId: admission?.scope.parentSessionId,
			}),
		);
	};

	/**
	 * Spawn a topic; with `seed`, the chosen messages of `seed.entries` are prepended to its first turn. The seed
	 * is built inside that turn (its failure fails the turn), under the admitted transcript's epoch: aborted, it
	 * cancels the summary request, the seed and the first turn, which is admitted only while the epoch is live.
	 * A seeded topic whose transcript was replaced by the time it is spawned is ended and refused.
	 */
	const spawnTopic = async (args: {
		title: string;
		message: string;
		name?: string;
		seed?: { admission: Admission; entries: readonly SessionEntry[]; messageIds: string[]; reason?: string };
	}): Promise<StartedTopic> => {
		const title = args.title.trim();
		if (!title) throw new Error("Topic title must not be empty.");
		const seed = args.seed;
		const admitted = Promise.withResolvers<void>();
		// Observed by the caller later; never an unhandled rejection meanwhile.
		admitted.promise.catch(() => {});
		if (!seed) admitted.resolve();
		const signal = seed?.admission.epoch.signal;
		const { id, jobId } = await registry.spawn(session, {
			cli: "good",
			name: args.name ?? title,
			prompt: args.message,
			title,
			...(seed && signal
				? {
						seed: {
							build: async () => {
								try {
									throwIfStale(signal);
									const origin = { sessionId: session.getSessionId?.() ?? "", signal };
									const text = await buildTopicSeed({
										entries: seed.entries,
										messageIds: seed.messageIds,
										reason: seed.reason,
										summarize: transcript => summarize(transcript, origin),
									});
									throwIfStale(signal);
									return text;
								} catch (error) {
									admitted.reject(error);
									throw error;
								}
							},
							dispatch: () => {
								try {
									throwIfStale(signal);
								} catch (error) {
									admitted.reject(error);
									throw error;
								}
								admitted.resolve();
							},
						},
					}
				: {}),
		});
		// A first turn that fails before it dispatches (its options, its lifecycle write) settles `admitted` too.
		const turn = session.asyncJobManager?.getJob(jobId)?.promise;
		turn?.catch(error => admitted.reject(error));
		if (signal?.aborted) {
			void endTopic(id, seed?.admission, turn);
			throw new StaleTranscriptError();
		}
		return { id, admitted: admitted.promise };
	};

	/**
	 * Once `started`'s first turn is admitted under the admitted transcript's live epoch, run `commit` (the accepted
	 * state, the note). A cancelled or failed first turn, or a failed commit, ends the topic and commits nothing.
	 */
	const afterAdmission = async (admission: Admission, started: StartedTopic, commit: () => Promise<void>) => {
		try {
			await started.admitted;
			await commit();
		} catch (error) {
			logger.debug("topics: topic not linked; its first turn was cancelled or failed", {
				id: started.id,
				error: error instanceof Error ? error.message : String(error),
			});
			void endTopic(started.id, admission, undefined);
			throw error;
		}
	};

	/** The visible "Continued in topic" note, in the admitted transcript only. */
	const noteContinued = ({ box, epoch }: Admission, details: TopicContinuedDetails): Promise<boolean> =>
		box.appendNote(epoch, {
			customType: TOPIC_CONTINUED_TYPE,
			content: `Continued in topic ${details.title}`,
			details,
		});

	/**
	 * Accept `suggestion` of the admitted transcript: spawn its topic and return its id at once; the accepted
	 * state and the note follow once its first turn is admitted. Until then the accept is pending in memory (a
	 * repeat joins it); if the first turn is cancelled or fails, the suggestion stays open and the topic is ended.
	 */
	const accept = async (admission: Admission, suggestion: TopicSuggestion, by: string): Promise<string> => {
		const started = await spawnTopic({
			title: suggestion.title,
			message: prompt.render(topicAcceptedTemplate, { title: suggestion.title }).trim(),
			seed: {
				admission,
				entries: [...admission.box.branch()],
				messageIds: suggestion.messageIds,
				reason: suggestion.reason,
			},
		}).catch(translateStale(suggestion.id));
		const topicId = started.id;
		void afterAdmission(admission, started, async () => {
			await commitState(admission, { id: suggestion.id, state: "accepted", topicId, by, atMs: Date.now() });
			await noteContinued(admission, {
				topicId,
				title: suggestion.title,
				suggestionId: suggestion.id,
				messageIds: [...suggestion.messageIds],
			});
		})
			.catch(() => {})
			.finally(() => {
				if (accepting.get(suggestion.id)?.epochId === admission.epoch.id) accepting.delete(suggestion.id);
			});
		return topicId;
	};

	return {
		list: () => registry.describe(session),
		async create({ title, message, name, seedMessageIds }) {
			if (!seedMessageIds?.length) return { id: (await spawnTopic({ title, message, name })).id };
			const admission = admit();
			const entries = admission.box.branch();
			requireEligibleMessageIds(seedMessageIds, entries);
			const started = await spawnTopic({
				title,
				message,
				name,
				seed: { admission, entries: [...entries], messageIds: seedMessageIds },
			});
			void afterAdmission(admission, started, async () => {
				await noteContinued(admission, {
					topicId: started.id,
					title: title.trim(),
					suggestionId: null,
					messageIds: [...seedMessageIds],
				});
			}).catch(() => {});
			return { id: started.id };
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
			const epochId = openEpochId(inbox);
			const eligible = eligibleIds(inbox);
			return [...folded(inbox, epochId).values()]
				.filter(suggestion => offered(suggestion, eligible, epochId))
				.map(suggestionSummary);
		},
		suggestionBoard() {
			if (!inbox) return { open: [], dismissed: [] };
			const epochId = openEpochId(inbox);
			const eligible = eligibleIds(inbox);
			const all = [...folded(inbox, epochId).values()];
			return {
				open: all.filter(suggestion => offered(suggestion, eligible, epochId)).map(suggestionSummary),
				dismissed: all
					.filter(suggestion => suggestion.state === "dismissed")
					.sort((a, b) => (b.stateOrder ?? 0) - (a.stateOrder ?? 0))
					.slice(0, BOARD_DISMISSED_TITLES)
					.map(suggestion => suggestion.title),
			};
		},
		suggest({ title, reason, messageIds }) {
			const admission = admit();
			const { box, epoch } = admission;
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
			const pending = { epochId: epoch.id, suggestion };
			unsaved.push(pending);
			void box
				.appendNote(epoch, {
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
			const suggestion = folded(admission.box, admission.epoch.id).get(suggestionId);
			if (!suggestion) throw new TopicsError("suggestion_not_found", `Unknown suggestion: ${suggestionId}`);
			if (suggestion.state === "accepted" && suggestion.topicId) {
				return { topicId: suggestion.topicId, alreadyAccepted: true };
			}
			const eligible = eligibleIds(admission.box);
			if (!suggestion.messageIds.every(id => eligible.has(id))) {
				throw new TopicsError(
					"suggestion_not_found",
					`Suggestion ${suggestionId} names messages this conversation no longer has (cleared).`,
				);
			}
			const operation = accept(admission, suggestion, by);
			accepting.set(suggestionId, { epochId: admission.epoch.id, promise: operation });
			try {
				return { topicId: await operation, alreadyAccepted: false };
			} catch (error) {
				accepting.delete(suggestionId);
				throw error;
			}
		},
		async dismissSuggestion({ suggestionId, by }) {
			const admission = admit();
			const running = inFlight(dismissing, admission, suggestionId);
			if (running) {
				await running;
				return { alreadyDismissed: true };
			}
			const suggestion = folded(admission.box, admission.epoch.id).get(suggestionId);
			if (!suggestion) throw new TopicsError("suggestion_not_found", `Unknown suggestion: ${suggestionId}`);
			if (suggestion.state === "accepted" || accepting.get(suggestionId)?.epochId === admission.epoch.id) {
				throw new TopicsError("already_accepted", `Suggestion ${suggestionId} was already accepted.`);
			}
			// The journal keeps only saved states: a dismissal whose write failed left nothing, so it is retried.
			if (suggestion.state === "dismissed") return { alreadyDismissed: true };
			const operation = commitState(admission, { id: suggestionId, state: "dismissed", by, atMs: Date.now() });
			dismissing.set(suggestionId, { epochId: admission.epoch.id, promise: operation });
			try {
				await operation;
				return { alreadyDismissed: false };
			} finally {
				dismissing.delete(suggestionId);
			}
		},
		async organise({ by }) {
			const { box, epoch } = admit();
			if (box.turnPending(TOPIC_ORGANISE_TYPE)) {
				throw new TopicsError("organise_running", "An Organise turn is already queued or running.");
			}
			// Rendered right before dispatch; the session refuses or drops the turn once its transcript is replaced.
			return box.queueTurn(epoch, TOPIC_ORGANISE_TYPE, () => {
				const index = formatMessageIndex(box.branch(), ORGANISE_INDEX_MESSAGES);
				return {
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
