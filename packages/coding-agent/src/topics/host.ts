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
import { buildTopicSeed, type SeedSummarizer, shortestTopicSeed, smolSeedSummarizer, type TopicSeed } from "./seed";
import { requireSeedRoom } from "./seed-room";
import { cfgTopicsEnabled } from "./settings";
import {
	eligibleMessageIds,
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
	TOPIC_START_FAILED_TYPE,
	TOPIC_SUGGESTION_TYPE,
	type TopicContinuedDetails,
	TopicsError,
	type TopicSuggestion,
	type TopicSuggestionStateData,
	type TopicStartFailedDetails,
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

/**
 * A seeded topic that could not start: even its shortest seed left its first request (`tokens`) over its model's
 * limit (`limit`), so that turn was refused. Sent once, after the topic ended and the Inbox note was written; `id`
 * is the topic's, `suggestionId` the accepted suggestion's (open again; absent for `create`).
 */
export interface StartFailedChange {
	id: string;
	ownerId: string;
	parentSessionId: string;
	reason: "start-failed";
	suggestionId?: string;
	code: "seed_too_large";
	tokens: number;
	limit: number;
}

/** What `subscribe` streams to a host: a topic's change, a suggestion's, or a seeded topic that could not start. */
export type TopicChange = WorkerChange | SuggestionChange | StartFailedChange;

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

/**
 * A spawned topic, whether the child admitted its first prompt (the runtime reports it on every exit), and the
 * size refusal of that prompt, if its seed could not fit.
 */
interface StartedTopic {
	id: string;
	firstTurn: Promise<"admitted" | "refused">;
	refusal(): { tokens: number; limit: number } | undefined;
}

/** The vibe cli a topic's child runs as. */
const TOPIC_CLI = "good";

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
	/** Forget `operation` on `suggestionId` unless a newer one (another epoch's) replaced it meanwhile. */
	const settleInFlight = <T>(map: Map<string, InFlight<T>>, suggestionId: string, operation: InFlight<T>) => {
		if (map.get(suggestionId) === operation) map.delete(suggestionId);
	};

	/** The suggestions of the open transcript: its journal, plus those still being recorded in epoch `epochId`. */
	const folded = (box: TopicsInbox, epochId: number | undefined) =>
		foldSuggestions(
			box.entries(),
			unsaved.filter(entry => entry.epochId === epochId).map(entry => entry.suggestion),
		);
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
	 * cancels the summary request, the seed and the child's launch, whose first prompt is admitted only while the
	 * epoch is live (the child saves and sends nothing of it otherwise).
	 * A seeded topic whose transcript was replaced by the time it is spawned is ended and refused. One whose
	 * shortest seed fits no model the child can use is refused before it is spawned (`seed_too_large`).
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
		if (seed) {
			requireSeedRoom(session, registry.workerModelPatterns(session, TOPIC_CLI), {
				task: args.message,
				seed: shortestTopicSeed(seed),
				overheadTokens: seed.admission.box.requestOverheadTokens?.() ?? 0,
			});
		}
		let built: TopicSeed | undefined;
		const firstTurn = Promise.withResolvers<"admitted" | "refused">();
		if (!seed) firstTurn.resolve("admitted");
		const signal = seed?.admission.epoch.signal;
		const { id, jobId } = await registry.spawn(session, {
			cli: TOPIC_CLI,
			name: args.name ?? title,
			prompt: args.message,
			title,
			...(seed && signal
				? {
						seed: {
							build: async () => {
								throwIfStale(signal);
								const origin = { sessionId: session.getSessionId?.() ?? "", signal };
								built = await buildTopicSeed({
									entries: seed.entries,
									messageIds: seed.messageIds,
									reason: seed.reason,
									summarize: transcript => summarize(transcript, origin),
								});
								throwIfStale(signal);
								return built;
							},
							signal,
							firstTurn: firstTurn.resolve,
						},
					}
				: {}),
		});
		if (signal?.aborted) {
			void endTopic(id, seed?.admission, session.asyncJobManager?.getJob(jobId)?.promise);
			throw new StaleTranscriptError();
		}
		return { id, firstTurn: firstTurn.promise, refusal: () => built?.refusal };
	};

	/**
	 * Tell the admitted transcript (a visible note), then the host (a `start-failed` change), that the seeded topic
	 * `topic` could not start: even its shortest seed did not fit its first request.
	 */
	const reportStartFailed = async (
		admission: Admission,
		topic: { id: string; title: string; suggestionId: string | null },
		refusal: { tokens: number; limit: number },
	) => {
		const details: TopicStartFailedDetails = {
			topicId: topic.id,
			title: topic.title,
			suggestionId: topic.suggestionId,
			code: "seed_too_large",
			...refusal,
		};
		const reopened = topic.suggestionId ? " The suggestion is open again." : "";
		await admission.box
			.appendNote(admission.epoch, {
				customType: TOPIC_START_FAILED_TYPE,
				content: `Topic "${topic.title}" could not start: the copied history is too large for the available models (${refusal.tokens} tokens, limit ${refusal.limit}).${reopened}`,
				details,
			})
			.catch(() => false);
		const change: StartFailedChange = {
			id: topic.id,
			ownerId: admission.scope.ownerId,
			parentSessionId: admission.scope.parentSessionId,
			reason: "start-failed",
			...(topic.suggestionId ? { suggestionId: topic.suggestionId } : {}),
			code: "seed_too_large",
			...refusal,
		};
		for (const listener of suggestionListeners) listener(change);
	};

	/**
	 * Once `started`'s first turn is admitted under the admitted transcript's live epoch, run `commit` (the accepted
	 * state, the note). A cancelled or failed first turn, or a failed commit, ends the topic and commits nothing; a
	 * first turn refused because its seed did not fit is reported ({@link reportStartFailed}) once the topic is ended
	 * and `topic.release` dropped what still holds it (a pending accept), so a host acting on the report finds the
	 * topic ended and the suggestion open.
	 */
	const afterAdmission = async (
		admission: Admission,
		started: StartedTopic,
		topic: { title: string; suggestionId: string | null; release?: () => void },
		commit: () => Promise<void>,
	) => {
		try {
			if ((await started.firstTurn) !== "admitted") throw new Error("The topic's first turn was not admitted.");
			await commit();
		} catch (error) {
			logger.debug("topics: topic not linked; its first turn was cancelled or failed", {
				id: started.id,
				error: error instanceof Error ? error.message : String(error),
			});
			const refusal = started.refusal();
			if (!refusal) {
				void endTopic(started.id, admission, undefined);
				throw error;
			}
			await endTopic(started.id, admission, undefined);
			topic.release?.();
			await reportStartFailed(
				admission,
				{ id: started.id, title: topic.title, suggestionId: topic.suggestionId },
				refusal,
			);
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
	 * `entries` is the branch its eligibility was checked on. The pending accept is registered before anything
	 * starts and forgotten by this same owner, once the spawn failed or the topic is linked or not.
	 */
	const accept = (
		admission: Admission,
		suggestion: TopicSuggestion,
		by: string,
		entries: readonly SessionEntry[],
	): Promise<string> => {
		const topic = Promise.withResolvers<string>();
		const operation: InFlight<string> = { epochId: admission.epoch.id, promise: topic.promise };
		accepting.set(suggestion.id, operation);
		void (async () => {
			try {
				const started = await spawnTopic({
					title: suggestion.title,
					message: prompt.render(topicAcceptedTemplate, { title: suggestion.title }).trim(),
					seed: {
						admission,
						entries: [...entries],
						messageIds: suggestion.messageIds,
						reason: suggestion.reason,
					},
				}).catch(translateStale(suggestion.id));
				const topicId = started.id;
				topic.resolve(topicId);
				const release = () => settleInFlight(accepting, suggestion.id, operation);
				await afterAdmission(
					admission,
					started,
					{ title: suggestion.title, suggestionId: suggestion.id, release },
					async () => {
						await commitState(admission, { id: suggestion.id, state: "accepted", topicId, by, atMs: Date.now() });
						await noteContinued(admission, {
							topicId,
							title: suggestion.title,
							suggestionId: suggestion.id,
							messageIds: [...suggestion.messageIds],
						});
					},
				);
			} catch (error) {
				// The spawn failed (the caller is refused); a link that failed after it was logged by `afterAdmission`.
				topic.reject(error);
			} finally {
				settleInFlight(accepting, suggestion.id, operation);
			}
		})();
		return topic.promise;
	};

	return {
		list: () => registry.describe(session),
		async create({ title, message, name, seedMessageIds }) {
			if (!seedMessageIds?.length) return { id: (await spawnTopic({ title, message, name })).id };
			const admission = admit();
			const entries = admission.box.branch();
			requireEligibleMessageIds(seedMessageIds, eligibleMessageIds(entries));
			const started = await spawnTopic({
				title,
				message,
				name,
				seed: { admission, entries: [...entries], messageIds: seedMessageIds },
			});
			void afterAdmission(admission, started, { title: title.trim(), suggestionId: null }, async () => {
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
			const eligible = eligibleMessageIds(inbox.branch());
			return [...folded(inbox, epochId).values()]
				.filter(suggestion => offered(suggestion, eligible, epochId))
				.map(suggestionSummary);
		},
		suggestionBoard() {
			if (!inbox) return { open: [], dismissed: [] };
			const epochId = openEpochId(inbox);
			const eligible = eligibleMessageIds(inbox.branch());
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
				requireEligibleMessageIds(messageIds, eligibleMessageIds(branch));
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
			const joined = inFlight(accepting, admission, suggestionId);
			if (joined) return { topicId: await joined, alreadyAccepted: true };
			const suggestion = folded(admission.box, admission.epoch.id).get(suggestionId);
			if (!suggestion) throw new TopicsError("suggestion_not_found", `Unknown suggestion: ${suggestionId}`);
			if (suggestion.state === "accepted" && suggestion.topicId) {
				return { topicId: suggestion.topicId, alreadyAccepted: true };
			}
			const entries = admission.box.branch();
			const eligible = eligibleMessageIds(entries);
			if (!suggestion.messageIds.every(id => eligible.has(id))) {
				throw new TopicsError(
					"suggestion_not_found",
					`Suggestion ${suggestionId} names messages this conversation no longer has (cleared).`,
				);
			}
			return { topicId: await accept(admission, suggestion, by, entries), alreadyAccepted: false };
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
			const operation: InFlight<void> = {
				epochId: admission.epoch.id,
				promise: commitState(admission, { id: suggestionId, state: "dismissed", by, atMs: Date.now() }),
			};
			dismissing.set(suggestionId, operation);
			try {
				await operation.promise;
				return { alreadyDismissed: false };
			} finally {
				settleInFlight(dismissing, suggestionId, operation);
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
