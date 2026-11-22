/**
 * What the topics host needs from the Inbox (the top-level session): its
 * entries, hidden state entries, visible notes that never start or interrupt a
 * turn, and one queued turn (Organise) that never interrupts a running one.
 * Every read and write goes through an {@link InboxBinding}: it refuses once
 * another transcript is open, so nothing meant for one conversation lands in the next.
 */
import { logger } from "@oh-my-pi/pi-utils";
import type { AgentSession } from "../session/agent-session";
import type { SessionEntry } from "../session/session-entries";
import { InboxBinding } from "./binding";

/** A custom message for the Inbox. */
export interface InboxMessage {
	customType: string;
	content: string;
	details: unknown;
}

export interface TopicsInbox {
	/** Bind to the open Inbox transcript. */
	bind(): InboxBinding;
	/** Entries of the bound transcript's current branch, root first; throws once it is not open. */
	branch(binding: InboxBinding): readonly SessionEntry[];
	/** Every entry of the bound transcript, in file order; throws once it is not open. */
	entries(binding: InboxBinding): readonly SessionEntry[];
	/**
	 * Append a hidden custom entry to the bound transcript, all or nothing: it
	 * resolves once the entry is on disk; a failed write leaves no entry behind
	 * and rejects, and so does another open transcript ({@link InboxChangedError},
	 * checked under the write lock, where `check` runs too).
	 */
	appendState(binding: InboxBinding, customType: string, data: unknown, check?: () => void): Promise<void>;
	/**
	 * Record a visible custom message in the bound transcript that the model sees
	 * from its next turn on ({@link AgentSession.recordWithoutTurn}): now while
	 * the Inbox is idle, else at its running turn's boundary. It never starts or
	 * interrupts a turn. Resolves true once it is on disk; false when another
	 * transcript is open, or it was dropped or could not be written.
	 */
	appendNote(binding: InboxBinding, message: InboxMessage): Promise<boolean>;
	/**
	 * Run `render()` (hidden) as one Inbox turn of the bound transcript, never
	 * overtaking or interrupting other work: behind prompts still in setup, then
	 * started at once when idle, else queued as a follow-up. `render` runs right
	 * before dispatch, once the bound transcript is confirmed open; another open
	 * transcript rejects with {@link InboxChangedError} and dispatches nothing.
	 * Resolves once the turn started (`queued: false`) or was queued (`queued: true`).
	 */
	queueTurn(binding: InboxBinding, customType: string, render: () => InboxMessage): Promise<{ queued: boolean }>;
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
	// Counts every session the Inbox adopts (new, fork, switch, reopen): a reopened session is a new transcript.
	let sessionEpoch = 0;
	session.registerSessionChangeCallback(() => {
		sessionEpoch++;
	});
	const identity = () => ({
		sessionId: manager.getSessionId(),
		sessionFile: manager.getSessionFile(),
		sessionEpoch,
		transcriptEpoch: session.transcriptEpoch,
	});

	return {
		bind: () => new InboxBinding(identity),
		branch(binding) {
			binding.assertCurrent();
			return manager.getBranch();
		},
		entries(binding) {
			binding.assertCurrent();
			return manager.getEntriesView();
		},
		async appendState(binding, customType, data, check) {
			binding.assertCurrent();
			await manager.appendEntriesAtomically(() => {
				// Checked again under the write lock: a transition may have happened while it waited.
				binding.assertCurrent();
				check?.();
				manager.appendCustomEntry(customType, data);
			});
		},
		async appendNote(binding, { customType, content, details }) {
			if (!binding.isCurrent) return false;
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
		async queueTurn(binding, customType, render) {
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
			let message: InboxMessage;
			try {
				// Right before dispatch: the bound transcript must still be the open one, and the message is rendered
				// from it, so nothing of a replaced transcript reaches the model or the journal.
				binding.assertCurrent();
				message = render();
			} catch (error) {
				turn.done();
				throw error;
			}
			const { content, details } = message;
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
