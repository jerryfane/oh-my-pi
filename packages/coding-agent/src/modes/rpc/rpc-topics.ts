/**
 * RPC `topics_*` commands: the host's direct line to the owner's topics. Every
 * command acts on the topic itself; none starts a turn of the main session.
 */
import type { TopicsHost } from "../../topics/host";
import type { RpcCommand } from "./rpc-types";

export type RpcTopicsCommand = Extract<RpcCommand, { type: `topics_${string}` }>;

/** Whether `command` is a `topics_*` command. */
export function isRpcTopicsCommand(command: RpcCommand): command is RpcTopicsCommand {
	return command.type.startsWith("topics_");
}

function requireText(value: unknown, field: string): string {
	if (typeof value !== "string" || !value.trim()) throw new Error(`\`${field}\` must be a non-empty string.`);
	return value;
}

/** Run one `topics_*` command; returns the response `data` (undefined = none). Throws a user-facing error. */
export async function runRpcTopicsCommand(host: TopicsHost, command: RpcTopicsCommand): Promise<object | undefined> {
	switch (command.type) {
		case "topics_list":
			return { topics: host.list() };
		case "topics_new": {
			const { id } = await host.create({
				title: requireText(command.title, "title"),
				message: requireText(command.message, "message"),
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
	}
}
