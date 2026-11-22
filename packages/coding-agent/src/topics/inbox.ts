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
	/** Append a hidden custom entry; resolves once it is on disk, rejects when the write fails. */
	appendState(customType: string, data: unknown): Promise<void>;
	/**
	 * Record a visible custom message that the model sees from its next turn on
	 * ({@link AgentSession.recordWithoutTurn}): now while the Inbox is idle, else
	 * at its running turn's boundary. It never starts or interrupts a turn.
	 * Resolves true once it is on disk; false when it was dropped (the session
	 * was disposed or its transcript replaced first) or could not be written.
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

interface QueuedTurn {
	/** The prompt call has not settled yet. */
	dispatching: boolean;
	/** The turn took the message (its `message_start` was seen). */
	started: boolean;
	done(): void;
}

/** The Inbox surface of a top-level {@link AgentSession}. */
export function sessionInbox(session: AgentSession): TopicsInbox {
	const turns = new Map<string, QueuedTurn>();

	return {
		sessionId: () => session.sessionManager.getSessionId(),
		branch: () => session.sessionManager.getBranch(),
		entries: () => session.sessionManager.getEntriesView(),
		async appendState(customType, data) {
			session.sessionManager.appendCustomEntry(customType, data);
			await session.sessionManager.flush();
		},
		async appendNote({ customType, content, details }) {
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
