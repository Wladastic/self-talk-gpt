import os
import time
import traceback
import torch
from pathlib import Path
from playsound import playsound
from dotenv import load_dotenv
import datetime
import requests

# Load environment variables from .env file.
load_dotenv()


base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:5000/")


def get_token_count(prompt):
    response = requests.post(base_url + "/api/v1/token-count", json={"prompt": prompt})
    if response.status_code == 200:
        return response.json()["results"][0]["tokens"]
    else:
        return 0


def trim_to_max_tokens(prompt, max_tokens):
    token_count = get_token_count(prompt)
    if token_count > (max_tokens):
        while token_count > (max_tokens):
            prompt = prompt.rsplit(".", 1)[0]
            token_count = get_token_count(prompt)
            # this is a hack to prevent infinite loops
            if token_count < max_tokens:
                break
    return prompt


def create_chat_completion(history, user_input, temperature, max_tokens, name):
    # print(temperature,max_tokens,type(temperature),type(max_tokens))
    if max_tokens is None:
        max_tokens = 2000
    if float(temperature) == 0.0:
        temperature = 0.5

    user_input = trim_to_max_tokens(user_input, max_tokens)

    # print("Sending request with token count: ", get_token_count(messages))
    request = {
        "user_input": user_input,
        "history": history,
        "mode": "chat",  # Valid options: 'chat', 'chat-instruct', 'instruct'
        "character": name,
        "instruction_template": "Wizard-Mega WizardLM",
        "your_name": name,
        "regenerate": False,
        "_continue": False,
        "stop_at_newline": False,
        "chat_prompt_size": 2048,
        "chat_generation_attempts": 1,
        "chat-instruct_command": "Continue the chat dialogue below.\n\n<|prompt|>",
        "max_new_tokens": max_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": 0.1,
        "typical_p": 1,
        "epsilon_cutoff": 0,  # In units of 1e-4
        "eta_cutoff": 0,  # In units of 1e-4
        "repetition_penalty": 1.18,
        "top_k": 40,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": False,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "seed": -1,
        "add_bos_token": True,
        "truncation_length": 2048,
        "ban_eos_token": False,
        "skip_special_tokens": True,
        "stopping_strings": [],
    }

    print(f"{name} is thinking...")
    response = requests.post(base_url + "/api/v1/chat", json=request)
    print(f"{name} is done thinking...")
    if response.status_code == 200:
        try:
            return response.json()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            exit(0)
    else:
        return "Error "


def create_completion(messages, temperature, max_tokens):
    # print(temperature,max_tokens,type(temperature),type(max_tokens))
    if max_tokens is None:
        max_tokens = 1000
    if float(temperature) == 0.0:
        temperature = 0.7
    request = {
        "prompt": str(messages),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    response = requests.post(base_url + "/api/v1/generate", json=request)

    if response.status_code == 200:
        resp = response.json()
        print(resp)
        resp = resp["results"][0]["text"].json()
        return resp[0]["content"]
    else:
        return "Error "


class TalkingAgent:
    def __init__(self, name, voice, temperature=0.7):
        # Initialize the TTS model and set the voice and speech_enabled attributes.
        self.name = name
        self._setup()
        self.voice = voice
        self.speech_enabled = os.getenv("ENABLE_SPEECH", "True").lower() in [
            "true",
            "1",
        ]
        prompt_prefix = os.getenv(
            "PROMPT_PREFIX",
            "This is a conversation with an AI. Please be kind and always respond in an efficient and meaningful way or make a suggestion.",
        )
        self.history = {"internal": [prompt_prefix], "visible": []}
        self.temperature = temperature

        # Rename the conversation log file with a timestamp
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        log_file_name = f"./conversations/conversation_log_{current_time}.txt"

        if os.path.isfile("./conversations/conversation_log.txt"):
            os.rename("./conversations/conversation_log.txt", log_file_name)

    def _setup(self):
        """Set up the TTS model."""

        print("Setting up...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            print(f"Generating speech for chunk {i + 1} of {len(chunks)}...")
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
        """Get the conversation log from the ./conversations/conversation_log.txt file."""
        with open("./conversations/conversation_log.txt", "r") as f:
            conversation_log = f.readlines()
        return [line.strip() for line in conversation_log]

    def generate_chat_response(self, prompt, max_tokens=2000):
        """Generate a response to a given prompt."""

        try:
            temperature = self.temperature
            max_tokens = max_tokens
            response = create_chat_completion(
                self.history, prompt, temperature, max_tokens, name=self.name
            )

            # print(f"{self.name} generated response: {response}")
            if not response:
                print("No response generated.")
            # remove any "\n" from the response
            # response_text = response.replace("\n", " ")
            # return response_text.strip()
            # print(response)
            result = response["results"][0]["history"]
            self.history = result

            return result["visible"][-1][1]

        except Exception as e:
            print(traceback.format_exc())
            print(f"Error generating response: {e}")
            exit(1)
            # return None

    def initialize_conversation(self, other_agent):
        """Initialize a conversation with another agent."""
        initial_prompt = os.getenv(
            "INITIAL_PROMPT",
            f"Hello {other_agent.name}, my name is {self.name}, how are you? ",
        )
        print("Initializing conversation...")
        # self._speech(initial_prompt)
        self.log_conversation(initial_prompt)

        prompt = initial_prompt
        max_attempts = 3  # Maximum number of attempts to generate a response
        self._speech(prompt)
        while max_attempts > 0:
            response = other_agent.generate_chat_response(prompt)
            if not response:
                max_attempts -= 1
                print(f"No response generated. {max_attempts} attempts remaining.")
                continue

            other_agent._speech(response)
            other_agent.log_conversation(response)

            # Pass the response to the other agent as the next prompt.
            prompt = self.generate_chat_response(response)
            if not prompt:
                print("Conversation ended.")
                break

            self._speech(prompt)
            self.log_conversation(prompt)

            max_attempts = 3  # Reset the maximum attempts if a response is generated

    def log_conversation(self, text):
        """Log the conversation to a text file."""

        print(f"Logging conversation: {text}")
        with open("./conversations/conversation_log.txt", "a") as f:
            f.write(self.name + ": " + text + "\n")


if __name__ == "__main__":
    # Fetch voice for each agent from environment variables.
    voice1 = os.getenv("VOICE1", "en_92")
    voice2 = os.getenv("VOICE2", "en_2")
    agent_1_name = os.getenv("AGENT_1_NAME", "Mona")
    agent_2_name = os.getenv("AGENT_2_NAME", "Jack")
    agent_1_temperature = float(os.getenv("AGENT_1_TEMPERATURE", 0.4))
    agent_2_temperature = float(os.getenv("AGENT_2_TEMPERATURE", 0.4))

    # Initialize two instances of TalkingAgent with different voices and names.
    agent_silero_tts1 = TalkingAgent(name=agent_1_name, voice=voice1, temperature=agent_1_temperature)
    agent_silero_tts2 = TalkingAgent(name=agent_2_name, voice=voice2, temperature=agent_2_temperature)

    # Start a conversation between the two agents.
    agent_silero_tts1.initialize_conversation(agent_silero_tts2)
