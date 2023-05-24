import os
import time
import torch
import openai
from pathlib import Path
from playsound import playsound
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file.
load_dotenv()


class TalkingAgent:
    def __init__(self, name, voice, temperature=0.5):
        # Initialize the TTS model and set the voice and speech_enabled attributes.
        self.name = name
        self._setup()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.voice = voice
        self.speech_enabled = os.getenv("ENABLE_SPEECH", "True").lower() in [
            "true",
            "1",
        ]
        self.temperature = temperature

        # Rename the conversation log file with a timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        log_file_name = f"conversation_log_{current_time}.txt"

        if os.path.isfile("conversation_log.txt"):
            os.rename("conversation_log.txt", log_file_name)

    def _setup(self):
        """Set up the TTS model."""

        print("Setting up...")
        device = torch.device("cpu")
        torch.set_num_threads(4)
        local_file = "model.pt"

        # Download the model if it doesn't exist locally.
        if not os.path.isfile(local_file):
            print("Downloading model...")
            torch.hub.download_url_to_file(
                "https://models.silero.ai/models/tts/en/v3_en.pt", local_file
            )

        # Load the model.
        print("Loading model...")
        self.model = torch.package.PackageImporter(local_file).load_pickle(
            "tts_models", "model"
        )
        self.model.to(device)
        print("Setup complete.")

    def _speech(self, text, voice_index=0):
        """Generate speech from text using the TTS model."""
        print(f"{self.name}: {text}")

        if not self.speech_enabled:
            print("Speech is disabled.")
            return

        # Replace common emoticons with words.
        text = text.replace(":)", "smiley face")

        sample_rate = 48000
        speaker = self.voice

        # Create output directory if it doesn't exist.
        output_dir = Path("./speech")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split the text into chunks of 1000 symbols or less.
        chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]

        # Generate speech for each chunk.
        for i, chunk in enumerate(chunks):
            # Save generated speech to a .wav file.
            output_file = output_dir / f"silero_{int(time.time())}_{i}.wav"

            audio = self.model.save_wav(
                text=chunk,
                speaker=speaker,
                sample_rate=sample_rate,
                audio_path=str(output_file),
            )

            # Play the generated speech.
            playsound(audio, True)

    def get_conversation_log(self):
        """Get the conversation log from the conversation_log.txt file."""
        with open("conversation_log.txt", "r") as f:
            conversation_log = f.readlines()
        return [line.strip() for line in conversation_log]

    def generate_response(self, prompt, max_tokens=2000, context=None):
        """Generate a response to a given prompt using OpenAI's GPT-3."""
        prompt_prefix = os.getenv(
            "PROMPT_PREFIX",
            "This is a conversation with an AI. Please be kind and always respond in an efficient and meaningful way or make a suggestion.",
        )
        prompt = f"{prompt_prefix}  {prompt}"

        if context is not None:
            # Add context to the prompt
            prompt = f"{context}\n{prompt}"

        print(f"{self.name} is thinking...")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are {self.name}."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            print(f"{self.name} generated response: {response}")
            if not response.choices[0].message["content"]:
                print("No response generated.")
            # remove any "\n" from the response
            response_text = response.choices[0].message["content"].replace("\n", " ")
            return response_text.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def initialize_conversation(self, other_agent):
        """Initialize a conversation with another agent."""
        initial_prompt = os.getenv(
            "INITIAL_PROMPT",
            f"Hello {other_agent.name}, my name is {self.name}, how are you?",
        )
        print("Initializing conversation...")
        self._speech(initial_prompt)
        self.log_conversation(initial_prompt)

        prompt = initial_prompt
        context = initial_prompt
        max_attempts = 3  # Maximum number of attempts to generate a response

        while max_attempts > 0:
            response = other_agent.generate_response(prompt, context=context)
            if not response:
                max_attempts -= 1
                print(f"No response generated. {max_attempts} attempts remaining.")
                continue

            other_agent._speech(response)
            other_agent.log_conversation(response)

            # Pass the response to the other agent as the next prompt.
            prompt = self.generate_response(response, context=context)
            if not prompt:
                print("Conversation ended.")
                break

            self._speech(prompt)
            self.log_conversation(prompt)

            # Update context with the last 10 messages
            conversation_log = self.get_conversation_log()
            context = "\n".join(
                conversation_log[-10:]
            )  # Get the last 10 messages as context
            max_attempts = 3  # Reset the maximum attempts if a response is generated

    def log_conversation(self, text):
        """Log the conversation to a text file."""

        print(f"Logging conversation: {text}")
        with open("conversation_log.txt", "a") as f:
            f.write(self.name + ": " + text + "\n")


if __name__ == "__main__":
    # Fetch voice for each agent from environment variables.
    voice1 = os.getenv("VOICE1", "en_92")
    voice2 = os.getenv("VOICE2", "en_2")

    # Initialize two instances of TalkingAgent with different voices and names.
    gpt3_silero_tts1 = TalkingAgent(name="Sophie", voice=voice1)
    gpt3_silero_tts2 = TalkingAgent(name="Jarvis", voice=voice2)

    # Start a conversation between the two agents.
    gpt3_silero_tts1.initialize_conversation(gpt3_silero_tts2)
