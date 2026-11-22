/**
 * What the topics host needs from the Inbox (the top-level session): its
 * entries, hidden state entries, visible notes that never start or interrupt a
 * turn, and one queued turn (Organise) that never interrupts a running one.
 */
import { logger } from "@oh-my-pi/pi-utils";
import type { AgentSession } from "../session/agent-session";
import type { CustomMessage } from "../session/messages";
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

/** The Inbox surface of a top-level {@link AgentSession}. */
export function sessionInbox(session: AgentSession): TopicsInbox {
	const notes: PendingNote[] = [];
	let flushing = false;
	const turns = new Map<string, QueuedTurn>();
	// An admitted submission is a prompt in setup or running: a note waits for it, so it never lands in a turn.
	const busy = (): boolean => session.isStreaming || session.isSessionTransitioning || session.hasAdmittedSubmission;

	/** The next session event of any kind. */
	const nextEvent = (): Promise<void> => {
		const { promise, resolve } = Promise.withResolvers<void>();
		const unsubscribe = session.subscribe(() => {
			unsubscribe();
			resolve();
		});
		return promise;
	};

	/** Wait until no turn, prompt or session transition holds the Inbox. */
	const untilIdle = async (): Promise<void> => {
		while (!session.isDisposed && busy()) {
			if (session.isSessionTransitioning) await session.waitForSessionTransition();
			else if (session.hasAdmittedSubmission) await session.waitForAdmittedSubmissions();
			else if (session.agent.state.isStreaming) await session.waitForIdle();
			// A turn in setup that no submission admitted (a wake, an auto-continue): wait for it to go live.
			else await nextEvent();
		}
	};

	/**
	 * Save every queued note once the Inbox is idle. The idle check and the
	 * appends run in one synchronous step, so no turn can start in between and
	 * take a note as a steer; `flushing` clears in that same step, so a caller
	 * woken by its note's promise starts the next flush itself.
	 */
	const flushNotes = async (): Promise<void> => {
		while (true) {
			if (!session.isDisposed && busy()) {
				await untilIdle();
				continue;
			}
			const current = session.isDisposed ? undefined : session.sessionManager.getSessionId();
			for (const note of notes.splice(0)) {
				if (note.sessionId !== current) {
					logger.warn("topics: dropped an Inbox note of a session no longer open", {
						customType: note.message.customType,
					});
					note.settle(false);
					continue;
				}
				const record: CustomMessage = {
					role: "custom",
					customType: note.message.customType,
					content: note.message.content,
					display: true,
					details: note.message.details,
					attribution: "agent",
					timestamp: Date.now(),
				};
				// Like an idle custom message with no turn: in the model's context from the next turn, and saved.
				session.agent.appendMessage(record);
				session.sessionManager.appendCustomMessageEntry(
					record.customType,
					record.content,
					record.display,
					record.details,
					record.attribution,
					record.timestamp,
				);
				note.settle(true);
			}
			if (notes.length === 0) {
				flushing = false;
				return;
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
			if (!flushing) {
				flushing = true;
				void flushNotes();
			}
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
