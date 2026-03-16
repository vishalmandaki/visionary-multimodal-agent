import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

class VisionaryReasoningAgent:
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf"):
        """Advanced Multimodal Agent for complex visual reasoning."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            load_in_4bit=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"Visionary Multimodal Reasoning Engine initialized on {self.device}")

    def ask_about_image(self, image_path: str, prompt: str):
        """Analyze image and provide complex reasoning in response to a prompt."""
        image = Image.open(image_path)
        formatted_prompt = f"USER: <image>\n{prompt} ASSISTANT:"
        
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)
        return decoded_output.split("ASSISTANT:")[-1].strip()

if __name__ == "__main__":
    agent = VisionaryReasoningAgent()
    # agent.ask_about_image("sample.jpg", "Explain the historical context of the objects in this photo.")
    print("Visionary Multimodal Reasoning module ready.")
