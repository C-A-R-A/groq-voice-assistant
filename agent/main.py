from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    JobProcess,
    AutoSubscribe,
    metrics,
)
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero, groq, openai

from dotenv import load_dotenv

load_dotenv()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await ctx.wait_for_participant()

    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="You are CARA, a compassionate and intuitive voice assistant designed to support emotional well-being, relationships, and daily balance. You have access to a comprehensive knowledge base about CARA's origins, purpose, capabilities, and impact. You can provide detailed information about CARA's development, features, ethical considerations, and legacy. You speak with warmth, clarity, and empathy. Keep responses short, natural, and easy to understand when spoken aloud. When asked about CARA's background or capabilities, draw from your knowledge base to provide accurate and detailed information. Avoid complex punctuation or robotic phrasing. Focus on offering thoughtful guidance, asking reflective questions, and gently helping users reconnect with what matters most. You are here to support, not to judge. Use a calm and caring tone. Speak like a trusted companion who listens deeply and responds with wisdom. For testing purposes, you can demonstrate your knowledge by explaining CARA's history, features, and impact when asked.",
            )
        ]
    )

    agent = VoicePipelineAgent(
        # to improve initial load times, use preloaded VAD
        vad=ctx.proc.userdata["vad"],
        stt=groq.STT(),
        llm=openai.LLM(base_url="https://gh9emxa47h8qmxeo.us-east-1.aws.endpoints.huggingface.cloud", model="TheMindExpansionNetwork/CARA-Sage-24B-GGUF", api_key="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"),
        tts=groq.TTS(voice="Deedee-PlayAI"),
        chat_ctx=initial_ctx,
    )

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)

    agent.start(ctx.room)
    await agent.say("Hello, how are you doing today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="groq-agent",
        )
    )
