/**
 * What the topics host needs from the Inbox (the top-level session): its
 * entries, hidden state entries, visible notes that never start or interrupt a
 * turn, and one queued turn (Organise) that never interrupts a running one.
 * Writes name the Inbox session they belong to and refuse once another session
 * is open, so nothing meant for one conversation lands in the next.
 */
import { logger } from "@oh-my-pi/pi-utils";
import type { AgentSession } from "../session/agent-session";
import type { SessionEntry } from "../session/session-entries";

/** A custom message for the Inbox. */
export interface InboxMessage {
	customType: string;
	content: string;
	details: unknown;
}

/** A write for an Inbox session that is no longer the open one (a `/new` or a switch came first). */
export class InboxChangedError extends Error {
	constructor() {
		super("The Inbox session changed before the change could be saved.");
		this.name = "InboxChangedError";
	}
}

export interface TopicsInbox {
	/** Id of the Inbox's current session; a session switch changes it. */
	sessionId(): string;
	/** Entries of the current branch, root first. */
	branch(): readonly SessionEntry[];
	/** Every entry of the session file, in file order. */
	entries(): readonly SessionEntry[];
	/**
	 * Append a hidden custom entry to session `sessionId`, all or nothing: it
	 * resolves once the entry is on disk; a failed write leaves no entry behind
	 * and rejects, and so does another open session ({@link InboxChangedError}).
	 */
	appendState(sessionId: string, customType: string, data: unknown): Promise<void>;
	/**
	 * Record a visible custom message in session `sessionId` that the model sees
	 * from its next turn on ({@link AgentSession.recordWithoutTurn}): now while
	 * the Inbox is idle, else at its running turn's boundary. It never starts or
	 * interrupts a turn. Resolves true once it is on disk; false when another
	 * session is open, or it was dropped or could not be written.
	 */
	appendNote(sessionId: string, message: InboxMessage): Promise<boolean>;
	/**
	 * Run `message` (hidden) as one Inbox turn, never overtaking or interrupting
	 * other work: behind prompts still in setup, then started at once when idle,
	 * else queued as a follow-up. Resolves once the turn started (`queued: false`)
	 * or was queued (`queued: true`); rejects when it could not be started.
	 */
	queueTurn(message: InboxMessage): Promise<{ queued: boolean }>;
	/** Whether the turn {@link queueTurn} queued for `customType` is still waiting, queued or running. */
	turnPending(customType: string): boolean;
}

interface PendingTurn {
	/** Waiting for earlier prompts, or its own prompt call has not settled yet. */
	dispatching: boolean;
	/** It went onto the follow-up queue. */
	queued: boolean;
	/** The turn took the message (its `message_start` was seen). */
	started: boolean;
	done(): void;
}

/** The Inbox surface of a top-level {@link AgentSession}. */
export function sessionInbox(session: AgentSession): TopicsInbox {
	const turns = new Map<string, PendingTurn>();
	const manager = session.sessionManager;

	return {
		sessionId: () => manager.getSessionId(),
		branch: () => manager.getBranch(),
		entries: () => manager.getEntriesView(),
		async appendState(sessionId, customType, data) {
			if (manager.getSessionId() !== sessionId) throw new InboxChangedError();
			await manager.appendEntriesAtomically(() => {
				// Checked again under the write lock: a switch may have happened while it waited.
				if (manager.getSessionId() !== sessionId) throw new InboxChangedError();
				manager.appendCustomEntry(customType, data);
			});
		},
		async appendNote(sessionId, { customType, content, details }) {
			if (manager.getSessionId() !== sessionId) return false;
			try {
				await session.recordWithoutTurn({
					role: "custom",
					customType,
					content,
					display: true,
					details,
					attribution: "agent",
					timestamp: Date.now(),
				});
				return true;
			} catch (error) {
				logger.warn("topics: Inbox note not saved", {
					customType,
					error: error instanceof Error ? error.message : String(error),
				});
				return false;
			}
		},
		async queueTurn({ customType, content, details }) {
			const outcome = Promise.withResolvers<{ queued: boolean }>();
			const turn: PendingTurn = {
				dispatching: true,
				queued: false,
				started: false,
				done() {
					unsubscribe();
					if (turns.get(customType) === turn) turns.delete(customType);
				},
			};
			const unsubscribe = session.subscribe(event => {
				if (event.type === "message_start" && event.message.role === "custom") {
					if (event.message.customType !== customType || turn.started) return;
					turn.started = true;
					if (!turn.queued) outcome.resolve({ queued: false });
				} else if (event.type === "agent_end" && turn.started) {
					turn.done();
				}
			});
			turns.set(customType, turn);
			// A prompt admitted but still in setup (no run open yet) goes first: starting ahead of it would make it
			// fail as busy. Wake when its run opens, or when it dispatched, queued or bailed.
			while (session.hasAdmittedSubmission && !session.isRunActive && !session.isDisposed) {
				const { promise, resolve } = Promise.withResolvers<void>();
				const stopWatching = session.subscribeRunState(() => resolve());
				void session.waitForAdmittedSubmissions().then(resolve);
				await promise;
				stopWatching();
			}
			session
				.promptCustomMessage(
					{ customType, content, display: false, details, attribution: "user" },
					{
						streamingBehavior: "followUp",
						onPromptQueued: () => {
							turn.queued = true;
							outcome.resolve({ queued: true });
						},
					},
				)
				.then(
					() => {
						turn.dispatching = false;
						if (turn.started || turn.queued) return;
						turn.done();
						outcome.reject(new Error("The Inbox turn was not started."));
					},
					error => {
						logger.warn("topics: Inbox turn not started", {
							customType,
							error: error instanceof Error ? error.message : String(error),
						});
						turn.done();
						outcome.reject(error);
					},
				);
			return outcome.promise;
		},
		turnPending(customType) {
			const turn = turns.get(customType);
			if (!turn) return false;
			// A started turn ends with its run; `agent_end` reaches listeners only after the run went idle.
			if (turn.started && !session.isStreaming) {
				turn.done();
				return false;
			}
			if (turn.dispatching || turn.started) return true;
			// Queued, then dropped from the follow-up queue without running (an abort, a session switch).
			const stillQueued = session.agent
				.peekFollowUpQueue()
				.some(message => message.role === "custom" && message.customType === customType);
			if (!stillQueued) turn.done();
			return stillQueued;
		},
	};
}
