/**
 * `topic_suggest`: the Inbox model suggests that a subject move into its own
 * topic. Registered only in the Inbox (top-level session) with topics enabled;
 * it records the suggestion and creates nothing.
 */
import { type } from "@oh-my-pi/omptype";
import type { AgentTool, AgentToolResult } from "@oh-my-pi/pi-agent-core";
import { prompt } from "@oh-my-pi/pi-utils";
import { ToolError } from "../tools/tool-errors";
import topicSuggestDescription from "../prompts/tools/topic-suggest.md" with { type: "text" };
import type { TopicsHost } from "./host";
import { type TopicSuggestion, topicsErrorCode } from "./suggestions";

export const TOPIC_SUGGEST_TOOL_NAME = "topic_suggest";

const topicSuggestSchema = type({
	title: type("string > 0").describe("short title of the topic, as the owner would name it"),
	reason: type("string > 0").describe("one sentence: why this subject deserves its own topic"),
	message_ids: type("string[]").describe(
		"session entry ids of every user and assistant message about this subject, and only those; exactly these are copied into the topic",
	),
});

export class TopicSuggestTool implements AgentTool<typeof topicSuggestSchema, TopicSuggestion> {
	readonly name = TOPIC_SUGGEST_TOOL_NAME;
	readonly approval = "read" as const;
	readonly label = "Topic Suggest";
	readonly summary = "Suggest moving a separate subject into its own topic";
	readonly description: string;
	readonly parameters = topicSuggestSchema;
	readonly strict = true;

	constructor(private readonly topics: () => Promise<TopicsHost | undefined>) {
		this.description = prompt.render(topicSuggestDescription);
	}

	async execute(
		_toolCallId: string,
		params: typeof topicSuggestSchema.infer,
	): Promise<AgentToolResult<TopicSuggestion>> {
		const host = await this.topics();
		if (!host) throw new ToolError("Topics are disabled.");
		let suggestion: TopicSuggestion;
		try {
			suggestion = host.suggest({ title: params.title, reason: params.reason, messageIds: params.message_ids });
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			const code = topicsErrorCode(error);
			throw new ToolError(code ? `${code}: ${message}` : message);
		}
		return {
			content: [
				{
					type: "text",
					text: `Suggested topic "${suggestion.title}" (${suggestion.id}, ${suggestion.messageIds.length} messages). The owner decides whether to start it.`,
				},
			],
			details: suggestion,
		};
	}
}
