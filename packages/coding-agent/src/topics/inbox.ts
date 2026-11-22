/**
 * What the topics host needs from the Inbox (the top-level session): its
 * entries, hidden state entries, visible notes that never start or interrupt a
 * turn, and one queued turn (Organise) that never interrupts a running one.
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

export interface TopicsInbox {
	/** Id of the Inbox's current session; a session switch changes it. */
	sessionId(): string;
	/** Entries of the current branch, root first. */
	branch(): readonly SessionEntry[];
	/** Every entry of the session file, in file order. */
	entries(): readonly SessionEntry[];
	/** Append a hidden custom entry now; the model never sees it. */
	appendState(customType: string, data: unknown): void;
	/**
	 * Save a visible custom message that the model sees from its next turn on:
	 * at once while the Inbox is idle, else once its running turn ends, so it
	 * never lands inside a turn. It never starts or interrupts a turn. Resolves
	 * true once saved; false when dropped (the session closed or switched first).
	 */
	appendNote(message: InboxMessage): Promise<boolean>;
	/**
	 * Run `message` (hidden) as one Inbox turn: started at once when idle, else
	 * queued as a follow-up behind the running turn, never interrupting it.
	 * `queued` = the Inbox was busy.
	 */
	queueTurn(message: InboxMessage): { queued: boolean };
	/** Whether the turn {@link queueTurn} queued for `customType` is still queued or running. */
	turnPending(customType: string): boolean;
}

interface PendingNote {
	sessionId: string;
	message: InboxMessage;
	settle(saved: boolean): void;
}

interface QueuedTurn {
	/** The prompt call has not settled yet. */
	dispatching: boolean;
	/** The turn took the message (its `message_start` was seen). */
	started: boolean;
	done(): void;
}

const IDLE_RECHECK_MS = 50;

/** The Inbox surface of a top-level {@link AgentSession}. */
export function sessionInbox(session: AgentSession): TopicsInbox {
	const notes: PendingNote[] = [];
	let flushing: Promise<void> | undefined;
	const turns = new Map<string, QueuedTurn>();
	const busy = (): boolean => session.isStreaming || session.isSessionTransitioning;

	const saveNotes = async (): Promise<void> => {
		while (notes.length > 0) {
			while (!session.isDisposed && busy()) {
				await session.waitForIdle();
				if (busy()) await Bun.sleep(IDLE_RECHECK_MS);
			}
			const note = notes.shift()!;
			if (session.isDisposed || note.sessionId !== session.sessionManager.getSessionId()) {
				logger.warn("topics: dropped an Inbox note of a session no longer open", {
					customType: note.message.customType,
				});
				note.settle(false);
				continue;
			}
			try {
				await session.sendCustomMessage({ ...note.message, display: true, attribution: "agent" });
				note.settle(true);
			} catch (error) {
				logger.warn("topics: Inbox note not saved", {
					customType: note.message.customType,
					error: error instanceof Error ? error.message : String(error),
				});
				note.settle(false);
			}
		}
	};

	return {
		sessionId: () => session.sessionManager.getSessionId(),
		branch: () => session.sessionManager.getBranch(),
		entries: () => session.sessionManager.getEntriesView(),
		appendState(customType, data) {
			session.sessionManager.appendCustomEntry(customType, data);
		},
		appendNote(message) {
			const { promise, resolve } = Promise.withResolvers<boolean>();
			notes.push({ sessionId: session.sessionManager.getSessionId(), message, settle: resolve });
			flushing ??= saveNotes().finally(() => {
				flushing = undefined;
			});
			return promise;
		},
		queueTurn({ customType, content, details }) {
			const queued = session.isStreaming;
			const turn: QueuedTurn = {
				dispatching: true,
				started: false,
				done() {
					unsubscribe();
					if (turns.get(customType) === turn) turns.delete(customType);
				},
			};
			const unsubscribe = session.subscribe(event => {
				if (event.type === "message_start" && event.message.role === "custom") {
					if (event.message.customType === customType) turn.started = true;
				} else if (event.type === "agent_end" && turn.started) {
					turn.done();
				}
			});
			turns.set(customType, turn);
			session
				.promptCustomMessage(
					{ customType, content, display: false, details, attribution: "user" },
					{ streamingBehavior: "followUp" },
				)
				.then(
					() => {
						turn.dispatching = false;
					},
					error => {
						logger.warn("topics: Inbox turn not started", {
							customType,
							error: error instanceof Error ? error.message : String(error),
						});
						turn.done();
					},
				);
			return { queued };
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
