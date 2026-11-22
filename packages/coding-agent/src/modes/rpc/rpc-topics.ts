/**
 * RPC `topics_*` commands: the host's direct line to the owner's topics. Every
 * command acts on the topic itself; none starts a turn of the main session,
 * except `topics_organise`, which queues one Inbox turn behind any running one.
 * A refusal with a machine-readable reason carries `code` next to `error`.
 */
import type { TopicsHost } from "../../topics/host";
import { type TopicSuggestion, type TopicsErrorCode, topicsErrorCode } from "../../topics/suggestions";
import type { RpcResponse, RpcSubagentMessagesResult } from "./rpc-types";

/** Owner-facing topics (persistent workers the host talks to directly); part of `RpcCommand`. */
export type RpcTopicsCommand =
	| { id?: string; type: "topics_list" }
	| { id?: string; type: "topics_new"; title: string; message: string; seedMessageIds?: string[] }
	| { id?: string; type: "topics_send"; topicId: string; message: string }
	| { id?: string; type: "topics_read"; topicId: string; fromByte?: number }
	| { id?: string; type: "topics_close"; topicId: string }
	| { id?: string; type: "topics_reopen"; topicId: string }
	| { id?: string; type: "topics_rename"; topicId: string; title: string }
	| { id?: string; type: "topics_accept_suggestion"; suggestionId: string; by: string }
	| { id?: string; type: "topics_dismiss_suggestion"; suggestionId: string; by: string }
	| { id?: string; type: "topics_organise"; by: string };

/** One owner-facing topic (persistent worker). */
export interface RpcTopicSummary {
	id: string;
	title: string;
	state: "starting" | "running" | "idle" | "dead" | "closed";
	turns: number;
	queued: number;
	createdAt: number;
	lastActivityAt: number;
	lastActivity?: string;
	/** First line of the owner's latest message to the topic, when this process saw one. */
	ownerLine?: string;
	sessionFile?: string;
	/** When the turn in flight started; absent while no turn runs. */
	turnStartedAt?: number;
}

/** One open topic suggestion of the Inbox model. */
export type RpcTopicSuggestion = TopicSuggestion;

/** Successful responses to {@link RpcTopicsCommand}; part of `RpcResponse`. */
export type RpcTopicsResponse =
	| {
			id?: string;
			type: "response";
			command: "topics_list";
			success: true;
			data: { topics: RpcTopicSummary[]; suggestions: RpcTopicSuggestion[] };
	  }
	| { id?: string; type: "response"; command: "topics_new"; success: true; data: { topicId: string } }
	| {
			id?: string;
			type: "response";
			command: "topics_send";
			success: true;
			data: { topicId: string; mode: "steered" | "queued" | "turn" };
	  }
	| { id?: string; type: "response"; command: "topics_read"; success: true; data: RpcSubagentMessagesResult }
	| { id?: string; type: "response"; command: "topics_close"; success: true }
	| { id?: string; type: "response"; command: "topics_reopen"; success: true }
	| { id?: string; type: "response"; command: "topics_rename"; success: true }
	| {
			id?: string;
			type: "response";
			command: "topics_accept_suggestion";
			success: true;
			data: { topicId: string; alreadyAccepted: boolean };
	  }
	| {
			id?: string;
			type: "response";
			command: "topics_dismiss_suggestion";
			success: true;
			data: { alreadyDismissed: boolean };
	  }
	| { id?: string; type: "response"; command: "topics_organise"; success: true; data: { queued: boolean } };

const TOPICS_COMMANDS: Record<RpcTopicsCommand["type"], true> = {
	topics_list: true,
	topics_new: true,
	topics_send: true,
	topics_read: true,
	topics_close: true,
	topics_reopen: true,
	topics_rename: true,
	topics_accept_suggestion: true,
	topics_dismiss_suggestion: true,
	topics_organise: true,
};

/** Whether `command` is one of the `topics_*` commands (a typo like `topics_clsoe` is not). */
export function isRpcTopicsCommand(command: { type: string }): command is RpcTopicsCommand {
	return Object.hasOwn(TOPICS_COMMANDS, command.type);
}

/** The refusal sent when topics are off, over RPC and the control socket. */
export const TOPICS_DISABLED: { error: string; code: TopicsErrorCode } = {
	error: "Topics are disabled; enable the topics.enabled setting.",
	code: "topics_disabled",
};

/** `error` and, for a refusal with a machine-readable reason, `code` of a failed `topics_*` command. */
export function topicsFailure(error: unknown): { error: string; code?: TopicsErrorCode } {
	const code = topicsErrorCode(error);
	return { error: error instanceof Error ? error.message : String(error), ...(code ? { code } : {}) };
}

function requireText(value: unknown, field: string): string {
	if (typeof value !== "string" || !value.trim()) throw new Error(`\`${field}\` must be a non-empty string.`);
	return value;
}

function optionalIds(value: unknown, field: string): string[] | undefined {
	if (value === undefined) return undefined;
	if (!Array.isArray(value) || !value.every(id => typeof id === "string" && id.length > 0)) {
		throw new Error(`\`${field}\` must be an array of non-empty strings.`);
	}
	return value;
}

/** Run one `topics_*` command; returns the response `data` (undefined = none). Throws a user-facing error. */
export async function runRpcTopicsCommand(host: TopicsHost, command: RpcTopicsCommand): Promise<object | undefined> {
	switch (command.type) {
		case "topics_list":
			return { topics: host.list(), suggestions: host.suggestions() };
		case "topics_new": {
			const { id } = await host.create({
				title: requireText(command.title, "title"),
				message: requireText(command.message, "message"),
				seedMessageIds: optionalIds(command.seedMessageIds, "seedMessageIds"),
			});
			return { topicId: id };
		}
		case "topics_send": {
			const outcome = await host.send(
				requireText(command.topicId, "topicId"),
				requireText(command.message, "message"),
			);
			return { topicId: outcome.id, mode: outcome.mode };
		}
		case "topics_read":
			if (command.fromByte !== undefined && !Number.isFinite(command.fromByte)) {
				throw new Error("`fromByte` must be a finite number.");
			}
			return host.read(requireText(command.topicId, "topicId"), command.fromByte);
		case "topics_close":
			await host.close(requireText(command.topicId, "topicId"));
			return undefined;
		case "topics_reopen":
			await host.reopen(requireText(command.topicId, "topicId"));
			return undefined;
		case "topics_rename":
			await host.rename(requireText(command.topicId, "topicId"), requireText(command.title, "title"));
			return undefined;
		case "topics_accept_suggestion":
			return host.acceptSuggestion({
				suggestionId: requireText(command.suggestionId, "suggestionId"),
				by: requireText(command.by, "by"),
			});
		case "topics_dismiss_suggestion":
			return host.dismissSuggestion({
				suggestionId: requireText(command.suggestionId, "suggestionId"),
				by: requireText(command.by, "by"),
			});
		case "topics_organise":
			return host.organise({ by: requireText(command.by, "by") });
	}
}

/** The RPC response to a `topics_*` command, run on the session's topics host (`undefined` = topics disabled). */
export async function respondRpcTopicsCommand(
	topics: Promise<TopicsHost | undefined>,
	command: RpcTopicsCommand,
): Promise<RpcResponse> {
	const { id, type } = command;
	const fail = (failure: { error: string; code?: string }): RpcResponse => ({
		id,
		type: "response",
		command: type,
		success: false,
		...failure,
	});
	const host = await topics;
	if (!host) return fail(TOPICS_DISABLED);
	try {
		const data = await runRpcTopicsCommand(host, command);
		return (
			data === undefined
				? { id, type: "response", command: type, success: true }
				: { id, type: "response", command: type, success: true, data }
		) as RpcResponse;
	} catch (err) {
		return fail(topicsFailure(err));
	}
}
