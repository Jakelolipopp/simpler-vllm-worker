import os
import sys
import traceback
import runpod
from runpod import RunPodLogger
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

log = RunPodLogger()

# Global vLLM engine instance
vllm_engine = None

def initialize_engine():
    """
    Initializes the vLLM engine synchronously. 
    Called lazily on the first request to ensure it attaches to the correct event loop.
    """
    global vllm_engine
    try:
        model_path = os.environ.get("MODEL_PATH", "/models/Qwen/Qwen3-0.6B")
        max_model_len = int(os.environ.get("MAX_MODEL_LEN", "32768"))
        gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"))
        
        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype="auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=False,
            enable_reasoning=True,
            reasoning_parser="deepseek_r1",
            trust_remote_code=False,
        )
        
        log.info(f"Initializing vLLM AsyncLLMEngine with model: {model_path}...")
        vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        log.info("vLLM engine initialized successfully.")
        
    except Exception as e:
        log.error(f"Failed to initialize vLLM engine: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)

async def handler(job):
    """
    RunPod serverless handler function.
    """
    global vllm_engine
    
    # Lazy initialization guarantees we are in the RunPod-managed event loop
    if vllm_engine is None:
        initialize_engine()

    try:
        job_input = job["input"]
        
        if "prompt" not in job_input:
            yield {"error": "Missing 'prompt' in input"}
            return
            
        prompt = job_input["prompt"]
        sampling_params = SamplingParams(
            temperature=job_input.get("temperature", 0.6),
            top_p=job_input.get("top_p", 0.95),
            top_k=job_input.get("top_k", 20),
            max_tokens=job_input.get("max_tokens", 32768),
        )
        
        request_id = job["id"]
        
        # Generate with vLLM
        results_generator = vllm_engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id
        )
        
        # Stream results back to RunPod
        async for request_output in results_generator:
            text = request_output.outputs[0].text
            yield {"output": text}
            
    except Exception as e:
        error_str = str(e)
        log.error(f"Error during inference: {error_str}")
        log.error(traceback.format_exc())
        
        if "CUDA" in error_str.lower() or "gpu" in error_str.lower():
            log.error("Terminating worker due to GPU error to trigger RunPod restart.")
            sys.exit(1)
            
        yield {"error": error_str}

if __name__ == "__main__":
    # Start the RunPod serverless worker
    # Engine is NO LONGER initialized here to protect the asyncio context
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True,
    })