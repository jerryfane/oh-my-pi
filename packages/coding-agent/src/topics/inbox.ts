/**
 * What the topics host needs from the Inbox (the top-level session): its
 * entries, hidden state entries, visible notes that never start or interrupt a
 * turn, and one queued turn (Organise) that never interrupts a running one.
 * Every write and turn carries an {@link InboxBinding}; the session checks it
 * where the effect happens, so nothing meant for one transcript lands in the next.
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
	/**
	 * The open transcript's conversation as the model has it: the current branch after its latest reset
	 * (`/clear`), root first. Message indexes and message-id checks use this.
	 */
	branch(): readonly SessionEntry[];
	/** Every entry of the open transcript, in file order (history before a reset included). */
	entries(): readonly SessionEntry[];
	/**
	 * Append a hidden custom entry to the bound transcript, all or nothing: it resolves once the entry is
	 * on disk; a failed write leaves no entry behind and rejects. Under the write lock the session refuses
	 * another open transcript ({@link InboxChangedError}) and `check` runs; either refusal writes nothing.
	 */
	appendState(binding: InboxBinding, customType: string, data: unknown, check?: () => void): Promise<void>;
	/**
	 * Record a visible custom message in the bound transcript that the model sees from its next turn on
	 * ({@link AgentSession.recordWithoutTurn}): now while the Inbox is idle, else at its running turn's
	 * boundary. It never starts or interrupts a turn. Resolves true once it is on disk; false when another
	 * transcript is open where it would be appended, or it was dropped or could not be written.
	 */
	appendNote(binding: InboxBinding, message: InboxMessage): Promise<boolean>;
	/**
	 * Run the hidden `customType` message `render()` returns as one Inbox turn of the bound transcript, never
	 * overtaking or interrupting other work: behind prompts still in setup, then started at once when idle,
	 * else queued as a follow-up. The session refuses another open transcript where the turn would start or
	 * queue ({@link InboxChangedError}): nothing is sent or saved. Resolves once the turn started
	 * (`queued: false`) or was queued (`queued: true`).
	 */
	queueTurn(
		binding: InboxBinding,
		customType: string,
		render: () => Pick<InboxMessage, "content" | "details">,
	): Promise<{ queued: boolean }>;
	/** Whether the turn {@link queueTurn} queued for `customType` is still waiting, queued or running. */
	turnPending(customType: string): boolean;
}

/** The entries after the latest reset boundary of `branch` (all of it when there is none). */
export function afterLatestReset(branch: readonly SessionEntry[]): readonly SessionEntry[] {
	const reset = branch.findLastIndex(entry => entry.type === "reset_boundary");
	return reset < 0 ? branch : branch.slice(reset + 1);
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
		// Bound during a transition, the transcript is provisional: such a binding matches none.
		bind: () =>
			new InboxBinding(
				session.isSessionTransitioning ? Number.NaN : session.transcriptGeneration,
				manager.getSessionId(),
				() => session.transcriptGeneration,
			),
		branch: () => afterLatestReset(manager.getBranch()),
		entries: () => manager.getEntriesView(),
		async appendState(binding, customType, data, check) {
			await session.appendEntriesAtomically(() => {
				check?.();
				manager.appendCustomEntry(customType, data);
			}, binding.generation);
		},
		async appendNote(binding, { customType, content, details }) {
			try {
				await session.recordWithoutTurn(
					{
						role: "custom",
						customType,
						content,
						display: true,
						details,
						attribution: "agent",
						timestamp: Date.now(),
					},
					binding.generation,
				);
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
			let rendered: Pick<InboxMessage, "content" | "details">;
			try {
				rendered = render();
			} catch (error) {
				turn.done();
				throw error;
			}
			const { content, details } = rendered;
			session
				.promptCustomMessage(
					{ customType, content, display: false, details, attribution: "user" },
					{
						streamingBehavior: "followUp",
						expectedGeneration: binding.generation,
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
