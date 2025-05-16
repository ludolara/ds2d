#!/usr/bin/env python3
import json
import time
import os
from typing import List, Dict, Tuple, Any, Optional
from shapely.geometry import Polygon
from tqdm import tqdm

# Import the Google Generative AI module
try:
    from google import generativeai as genai
except ImportError:
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Could not import Google GenerativeAI. Please install with 'pip install google-generativeai'")

# Import token estimation
try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("Could not import transformers. Please install with 'pip install transformers'")

# Initialize tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

def count_tokens(text: str) -> int:
    """
    Estimate token count using Llama-3 tokenizer as a rough approximation.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        Estimated token count
    """
    tokens = tokenizer.encode(text)
    return len(tokens)

def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract a valid JSON object from text.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON object or None if invalid
    """
    # Find potential JSON start
    start_idx = text.find("{")
    if start_idx < 0:
        return None
        
    # Try to parse increasingly larger substrings
    for end_idx in range(len(text), start_idx, -1):
        try:
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue
            
    return None


class SimpleFloorplanGenerator:
    """
    A simple generator for floor plans without validation or retry.
    """
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.7, 
        top_p: float = 0.95,
        max_tokens: int = 2048
    ):
        """
        Initialize the simple floor plan generator.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Gemini model name to use
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            max_tokens: Maximum tokens to generate
        """
        # Set up API key
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as argument or GOOGLE_API_KEY environment variable")
        
        # Initialize Gemini API
        genai.configure(api_key=self.api_key)
        
        # Create the model
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens
            )
        )
        
        # Statistics tracking
        self.stats = {
            "llm_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "overlap_count": 0,
            "invalid_geometry_count": 0
        }
        
        # Define system prompt for floor plan generation
        self.SYSTEM_PROMPT = """
You are a state-of-the-art floor-plan generator that translates JSON specifications and connectivity requirements defined by a bubble diagram into precise, optimized layouts. 
Your algorithm considers each room's dimensions, proportion, and desired adjacencies to produce an efficient arrangement that maximizes usable space while honoring all constraints.
Your top priority is that no two room polygons ever overlap. Rooms must be strictly disjoint, room interiors must never intersect.  
Your output must be a JSON object, where `output` key contains:
- `room_count`: the total number of room entries  
- `rooms`: a list of rooms. Each room entry in `rooms` must include:
 - `id`: formatted as `<room_type>|<unique_index>` (e.g. `"bedroom|2"` or `"bathroom|0"`)  
 - `room_type`: the room type (e.g. `"living_room"`, `"kitchen"`, etc.)
 - `area` in square meters (all positive numbers)  
 - `floor_polygon`: an ordered list of `{x: , y:}` vertices defining a simple polygon  
Additional rules:
- **Absolute non-overlap**: no two room polygons may share any interior point under any circumstances.
- Every `id` used in the bubble diagram must appear in the `rooms` list.  
Return only a JSON object containing an `output` key without extra commentary or explanation.
"""

    def generate_with_gemini(self, prompt: str, retry_on_limit: bool = True) -> str:
        """
        Generate a response using the Gemini API with rate limiting.
        
        Args:
            prompt: The input prompt
            retry_on_limit: Whether to retry on rate limits
            
        Returns:
            Generated text response
        """
        max_attempts = 3
        base_delay = 30  # seconds
        
        # Count input tokens
        input_tokens = count_tokens(prompt)
        self.stats["input_tokens"] += input_tokens
        
        for attempt in range(max_attempts):
            try:
                # Generate content using the Gemini API
                response = self.model.generate_content(prompt)
                self.stats["llm_call_count"] += 1  # Increment LLM call count
                
                # Extract the generated text
                output_text = response.text
                
                # Count output tokens
                output_tokens = count_tokens(output_text)
                self.stats["output_tokens"] += output_tokens
                
                # For total tokens, we only count new tokens (output tokens)
                # Input tokens are already accounted for in API cost
                self.stats["total_tokens"] += output_tokens
                
                return output_text
            except Exception as e:
                error_str = str(e)
                
                # Check if this is a rate limit error
                if "quota" in error_str.lower() or "429" in error_str:
                    # Extract retry delay if available
                    retry_delay = base_delay
                    if "retry_delay" in error_str:
                        try:
                            # Try to parse the retry delay from the error message
                            delay_part = error_str.split("retry_delay")[1].split("seconds:")[1].split("}")[0].strip()
                            retry_delay = int(delay_part)
                        except:
                            # If parsing fails, use base delay
                            pass
                    
                    if retry_on_limit and attempt < max_attempts - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return ""
                else:
                    # For other errors, don't retry
                    return ""
        
        return ""

    def generate_floor_plan(self, input_data: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a floor plan with a single API call.
        
        Args:
            input_data: Bubble diagram input
            
        Returns:
            Tuple of (generated floor plan JSON string, statistics dictionary)
        """
        # Reset statistics
        self.stats = {
            "llm_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "overlap_count": 0,
            "invalid_geometry_count": 0
        }
        
        # Build the prompt
        prompt = f"{self.SYSTEM_PROMPT}\n\nHere is the bubble diagram for the floor plan:\n\n{input_data}"
        
        try:
            # Generate floor plan
            result = self.generate_with_gemini(prompt)
            
            # Check for overlaps and invalid geometries in the generated plan
            try:
                # First, try to extract JSON from the result
                json_result = extract_json_from_text(result)
                
                if json_result:
                    # Use the check_for_overlaps helper function to identify issues
                    _, overlap_count, invalid_geometry_count = check_for_overlaps(json_result)
                    self.stats["overlap_count"] = overlap_count
                    self.stats["invalid_geometry_count"] = invalid_geometry_count
                else:
                    self.stats["overlap_count"] = -1
                    self.stats["invalid_geometry_count"] = -1
            except Exception as e:
                self.stats["overlap_count"] = -1
                self.stats["invalid_geometry_count"] = -1
            
            return result, self.stats
        except Exception as e:
            return "{}", self.stats


class RetryFloorplanGenerator:
    """
    A generator that retries from scratch if overlaps are detected.
    """
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.7, 
        top_p: float = 0.95,
        max_tokens: int = 2048,
        max_retries: int = 5
    ):
        """
        Initialize the retry floor plan generator.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Gemini model name to use
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts
        """
        # Set up API key
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as argument or GOOGLE_API_KEY environment variable")
        
        # Initialize Gemini API
        genai.configure(api_key=self.api_key)
        
        # Create the model
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens
            )
        )
        
        # Set max retries
        self.max_retries = max_retries
        
        # Statistics tracking
        self.stats = {
            "llm_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "retry_count": 0,
            "final_overlap_count": 0,
            "final_invalid_geometry_count": 0
        }
        
        # Define system prompt for floor plan generation
        self.SYSTEM_PROMPT = """
You are a state-of-the-art floor-plan generator that translates JSON specifications and connectivity requirements defined by a bubble diagram into precise, optimized layouts. 
Your algorithm considers each room's dimensions, proportion, and desired adjacencies to produce an efficient arrangement that maximizes usable space while honoring all constraints.
Your top priority is that no two room polygons ever overlap. Rooms must be strictly disjoint, room interiors must never intersect.  
Your output must be a JSON object, where `output` key contains:
- `room_count`: the total number of room entries  
- `rooms`: a list of rooms. Each room entry in `rooms` must include:
 - `id`: formatted as `<room_type>|<unique_index>` (e.g. `"bedroom|2"` or `"bathroom|0"`)  
 - `room_type`: the room type (e.g. `"living_room"`, `"kitchen"`, etc.)
 - `area` in square meters (all positive numbers)  
 - `floor_polygon`: an ordered list of `{x: , y:}` vertices defining a simple polygon  
Additional rules:
- **Absolute non-overlap**: no two room polygons may share any interior point under any circumstances.
- Every `id` used in the bubble diagram must appear in the `rooms` list.  
Return only a JSON object containing an `output` key without extra commentary or explanation.
"""

    def generate_with_gemini(self, prompt: str, retry_on_limit: bool = True) -> str:
        """
        Generate a response using the Gemini API with rate limiting.
        
        Args:
            prompt: The input prompt
            retry_on_limit: Whether to retry on rate limits
            
        Returns:
            Generated text response
        """
        max_attempts = 3
        base_delay = 30  # seconds
        
        # Count input tokens
        input_tokens = count_tokens(prompt)
        self.stats["input_tokens"] += input_tokens
        
        for attempt in range(max_attempts):
            try:
                # Generate content using the Gemini API
                response = self.model.generate_content(prompt)
                self.stats["llm_call_count"] += 1  # Increment LLM call count
                
                # Extract the generated text
                output_text = response.text
                
                # Count output tokens
                output_tokens = count_tokens(output_text)
                self.stats["output_tokens"] += output_tokens
                
                # For total tokens, we only count new tokens (output tokens)
                # Input tokens are tracked separately
                self.stats["total_tokens"] += output_tokens
                
                return output_text
            except Exception as e:
                error_str = str(e)
                
                # Check if this is a rate limit error
                if "quota" in error_str.lower() or "429" in error_str:
                    # Extract retry delay if available
                    retry_delay = base_delay
                    if "retry_delay" in error_str:
                        try:
                            # Try to parse the retry delay from the error message
                            delay_part = error_str.split("retry_delay")[1].split("seconds:")[1].split("}")[0].strip()
                            retry_delay = int(delay_part)
                        except:
                            # If parsing fails, use base delay
                            pass
                    
                    if retry_on_limit and attempt < max_attempts - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return ""
                else:
                    # For other errors, don't retry
                    return ""
        
        return ""

    def generate_floor_plan(self, input_data: str, min_delay: int = 10) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a floor plan with full retries on detection of overlaps.
        
        Args:
            input_data: Bubble diagram input
            min_delay: Minimum delay between API calls in seconds
            
        Returns:
            Tuple of (generated floor plan JSON string, statistics dictionary)
        """
        # Reset statistics
        self.stats = {
            "llm_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "retry_count": 0,
            "final_overlap_count": 0,
            "final_invalid_geometry_count": 0
        }
        
        # Build the prompt
        prompt = f"{self.SYSTEM_PROMPT}\n\nHere is the bubble diagram for the floor plan:\n\n{input_data}"
        
        best_result = ""
        best_overlaps = float('inf')
        best_invalid = float('inf')
        
        # Try generating up to max_retries times
        for retry in range(self.max_retries):
            # Generate floor plan
            result = self.generate_with_gemini(prompt)
            
            # Track retry count
            if retry > 0:
                self.stats["retry_count"] += 1
            
            # Check for overlaps and invalid geometries
            try:
                json_result = extract_json_from_text(result)
                
                if json_result:
                    _, overlap_count, invalid_geometry_count = check_for_overlaps(json_result)
                    
                    # If we found a perfect solution (no overlaps or invalid geometries), return it
                    if overlap_count == 0 and invalid_geometry_count == 0:
                        self.stats["final_overlap_count"] = 0
                        self.stats["final_invalid_geometry_count"] = 0
                        return result, self.stats
                    
                    # Otherwise, keep track of the best result so far
                    total_problems = overlap_count + invalid_geometry_count
                    if total_problems < (best_overlaps + best_invalid):
                        best_result = result
                        best_overlaps = overlap_count
                        best_invalid = invalid_geometry_count
            except Exception:
                pass
                
            # Add delay between retries if specified
            if retry < self.max_retries - 1 and min_delay > 0:
                time.sleep(min_delay)
        
        # Return the best result we found
        self.stats["final_overlap_count"] = best_overlaps
        self.stats["final_invalid_geometry_count"] = best_invalid
        
        return best_result, self.stats


# class IncrementalFloorplanGenerator:
#     """
#     An incremental generator that verifies rooms as they are added and rewinds when overlaps are detected.
#     """
    
#     def __init__(
#         self, 
#         api_key: str = None,
#         model: str = "gemini-1.5-pro",
#         temperature: float = 0.7, 
#         top_p: float = 0.95,
#         max_tokens: int = 2048,
#         max_iterations: int = 10
#     ):
#         """
#         Initialize the incremental floor plan generator.
        
#         Args:
#             api_key: Google API key (defaults to GOOGLE_API_KEY env var)
#             model: Gemini model name to use
#             temperature: Sampling temperature
#             top_p: Nucleus sampling probability
#             max_tokens: Maximum tokens to generate
#             max_iterations: Maximum number of incremental generations
#         """
#         # Set up API key
#         self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
#         if not self.api_key:
#             raise ValueError("API key must be provided either as argument or GOOGLE_API_KEY environment variable")
        
#         # Initialize Gemini API
#         genai.configure(api_key=self.api_key)
        
#         # Create the model
#         self.model = genai.GenerativeModel(
#             model_name=model,
#             generation_config=genai.GenerationConfig(
#                 temperature=temperature,
#                 top_p=top_p,
#                 max_output_tokens=max_tokens
#             )
#         )
        
#         # Set max iterations
#         self.max_iterations = max_iterations
        
#         # Statistics tracking
#         self.stats = {
#             "llm_call_count": 0,
#             "input_tokens": 0,
#             "output_tokens": 0,
#             "total_tokens": 0,
#             "iterations_count": 0,
#             "rewind_count": 0,
#             "final_overlap_count": 0,
#             "final_invalid_geometry_count": 0
#         }
        
#         # Define system prompt for floor plan generation
#         self.SYSTEM_PROMPT = """
# You are a state-of-the-art floor-plan generator that translates JSON specifications and connectivity requirements defined by a bubble diagram into precise, optimized layouts. 
# Your algorithm considers each room's dimensions, proportion, and desired adjacencies to produce an efficient arrangement that maximizes usable space while honoring all constraints.
# Your top priority is that no two room polygons ever overlap. Rooms must be strictly disjoint, room interiors must never intersect.  
# Your output must be a JSON object, where `output` key contains:
# - `room_count`: the total number of room entries  
# - `rooms`: a list of rooms. Each room entry in `rooms` must include:
#  - `id`: formatted as `<room_type>|<unique_index>` (e.g. `"bedroom|2"` or `"bathroom|0"`)  
#  - `room_type`: the room type (e.g. `"living_room"`, `"kitchen"`, etc.)
#  - `area` in square meters (all positive numbers)  
#  - `floor_polygon`: an ordered list of `{x: , y:}` vertices defining a simple polygon  
# Additional rules:
# - **Absolute non-overlap**: no two room polygons may share any interior point under any circumstances.
# - Every `id` used in the bubble diagram must appear in the `rooms` list.  
# Return only a JSON object containing an `output` key without extra commentary or explanation.
# """

#     def generate_with_gemini(self, prompt: str, retry_on_limit: bool = True) -> str:
#         """
#         Generate a response using the Gemini API with rate limiting.
        
#         Args:
#             prompt: The input prompt
#             retry_on_limit: Whether to retry on rate limits
            
#         Returns:
#             Generated text response
#         """
#         max_attempts = 3
#         base_delay = 30  # seconds
        
#         # Count input tokens
#         input_tokens = count_tokens(prompt)
#         self.stats["input_tokens"] += input_tokens
        
#         for attempt in range(max_attempts):
#             try:
#                 # Generate content using the Gemini API
#                 response = self.model.generate_content(prompt)
#                 self.stats["llm_call_count"] += 1  # Increment LLM call count
                
#                 # Extract the generated text
#                 output_text = response.text
                
#                 # Count output tokens
#                 output_tokens = count_tokens(output_text)
#                 self.stats["output_tokens"] += output_tokens
                
#                 # For total tokens, we only count new tokens (output tokens)
#                 # Input tokens are tracked separately
#                 self.stats["total_tokens"] += output_tokens
                
#                 return output_text
#             except Exception as e:
#                 error_str = str(e)
                
#                 # Check if this is a rate limit error
#                 if "quota" in error_str.lower() or "429" in error_str:
#                     # Extract retry delay if available
#                     retry_delay = base_delay
#                     if "retry_delay" in error_str:
#                         try:
#                             # Try to parse the retry delay from the error message
#                             delay_part = error_str.split("retry_delay")[1].split("seconds:")[1].split("}")[0].strip()
#                             retry_delay = int(delay_part)
#                         except:
#                             # If parsing fails, use base delay
#                             pass
                    
#                     if retry_on_limit and attempt < max_attempts - 1:
#                         time.sleep(retry_delay)
#                         continue
#                     else:
#                         return ""
#                 else:
#                     # For other errors, don't retry
#                     return ""
        
#         return ""

#     def generate_floor_plan(self, input_data: str, min_delay: int = 10) -> Tuple[str, Dict[str, Any]]:
#         """
#         Generate a floor plan with intelligent rewind on overlap detection.
        
#         Args:
#             input_data: Bubble diagram input
#             min_delay: Minimum delay between API calls in seconds
            
#         Returns:
#             Tuple of (generated floor plan JSON string, statistics dictionary)
#         """
#         # Reset statistics
#         self.stats = {
#             "llm_call_count": 0,
#             "input_tokens": 0,
#             "output_tokens": 0,
#             "total_tokens": 0,
#             "iterations_count": 0,
#             "rewind_count": 0,
#             "final_overlap_count": 0,
#             "final_invalid_geometry_count": 0
#         }
        
#         # Build the initial prompt
#         initial_prompt = f"{self.SYSTEM_PROMPT}\n\nHere is the bubble diagram for the floor plan:\n\n{input_data}"
        
#         # Start with the first generation
#         current_result = self.generate_with_gemini(initial_prompt)
#         current_floor_plan = extract_json_from_text(current_result)
        
#         # If we couldn't get a valid initial JSON, return empty result
#         if not current_floor_plan:
#             return current_result, self.stats
            
#         # Track iterations
#         iterations = 0
        
#         # Create a structure to hold the valid rooms
#         valid_floor_plan = {
#             "output": {
#                 "room_count": 0,
#                 "rooms": []
#             }
#         }
        
#         # Create a list to track the valid polygons for overlap checking
#         valid_polygons = []
#         valid_room_ids = []
        
#         with tqdm(desc="Incremental generation", total=self.max_iterations, disable=True) as pbar:
#             while iterations < self.max_iterations:
#                 # Increment iteration count
#                 iterations += 1
#                 self.stats["iterations_count"] = iterations
                
#                 # Get the current rooms
#                 try:
#                     rooms = current_floor_plan["output"]["rooms"]
#                 except (KeyError, TypeError):
#                     # Try again with same prompt
#                     if min_delay > 0:
#                         time.sleep(min_delay)
#                     current_result = self.generate_with_gemini(initial_prompt)
#                     current_floor_plan = extract_json_from_text(current_result)
#                     if not current_floor_plan:
#                         break
#                     continue
                
#                 # Process each room incrementally to find the first problematic room
#                 last_valid_index = -1
#                 first_invalid_room = None
#                 first_invalid_index = -1
#                 should_rewind = False
                
#                 # Process each room one by one
#                 for room_idx, room in enumerate(rooms):
#                     # Skip rooms we've already validated and added to our valid floor plan
#                     if room_idx < len(valid_floor_plan["output"]["rooms"]):
#                         last_valid_index = room_idx
#                         continue
                    
#                     room_id = room.get("id", f"unknown|{room_idx}")
                    
#                     # Check if this room ID is already in our valid rooms
#                     if room_id in valid_room_ids:
#                         last_valid_index = room_idx
#                         continue
                    
#                     # Check if the room has a valid floor_polygon
#                     if "floor_polygon" not in room or not room["floor_polygon"] or len(room["floor_polygon"]) < 3:
#                         first_invalid_room = room
#                         first_invalid_index = room_idx
#                         should_rewind = True
#                         break
                    
#                     # Create a polygon from the room's floor_polygon
#                     try:
#                         # Extract coordinates, handling both y and z formats
#                         if "z" in room["floor_polygon"][0]:
#                             coords = [(p['x'], p['z']) for p in room["floor_polygon"]]
#                         else:
#                             coords = [(p['x'], p['y']) for p in room["floor_polygon"]]
                            
#                         # Create a polygon
#                         poly = Polygon(coords)
                        
#                         # Check if the polygon is valid
#                         if not poly.is_valid:
#                             first_invalid_room = room
#                             first_invalid_index = room_idx
#                             should_rewind = True
#                             break
                        
#                         # Check for overlaps with EXISTING valid polygons
#                         has_overlap = False
#                         for i, valid_poly in enumerate(valid_polygons):
#                             if poly.intersects(valid_poly) and not poly.touches(valid_poly):
#                                 first_invalid_room = room
#                                 first_invalid_index = room_idx
#                                 has_overlap = True
#                                 should_rewind = True
#                                 break
                                
#                         if has_overlap:
#                             break
                        
#                         # If we reach here, the room is valid - add it to our valid floor plan
#                         valid_floor_plan["output"]["rooms"].append(room)
#                         valid_polygons.append(poly)
#                         valid_room_ids.append(room_id)
#                         last_valid_index = room_idx
                        
#                     except (KeyError, ValueError):
#                         first_invalid_room = room
#                         first_invalid_index = room_idx
#                         should_rewind = True
#                         break
                
#                 # Update room count
#                 valid_floor_plan["output"]["room_count"] = len(valid_floor_plan["output"]["rooms"])
                
#                 # If we've processed all rooms without finding any issues, we're done
#                 if not should_rewind and last_valid_index >= len(rooms) - 1:
#                     break
                    
#                 # If we've reached the max iterations, stop
#                 if iterations >= self.max_iterations:
#                     break
                
#                 # If rewinding is needed, generate a continuation from our current valid state
#                 if should_rewind:
#                     self.stats["rewind_count"] += 1
                    
#                     # Create a partial floor plan with our valid rooms
#                     partial_floor_plan = {
#                         "output": {
#                             "room_count": len(valid_floor_plan["output"]["rooms"]),
#                             "rooms": valid_floor_plan["output"]["rooms"]
#                         }
#                     }
                    
#                     # Convert to JSON string
#                     partial_json = json.dumps(partial_floor_plan, indent=2)
                    
#                     # Create a new prompt with the input AND the valid partial output so far
#                     rewind_prompt = f"{self.SYSTEM_PROMPT}\n\nHere is the bubble diagram for the floor plan:\n\n{input_data}\n\n{partial_json}"
                    
#                     # Mandatory delay between API calls
#                     if min_delay > 0:
#                         time.sleep(min_delay)
                    
#                     # Generate a continuation
#                     continuation_result = self.generate_with_gemini(rewind_prompt)
#                     continuation_floor_plan = extract_json_from_text(continuation_result)
                    
#                     # If we got a valid continuation, use it for the next iteration
#                     if continuation_floor_plan:
#                         current_floor_plan = continuation_floor_plan
#                     else:
#                         # If the continuation wasn't valid, try again with the initial prompt
#                         if min_delay > 0:
#                             time.sleep(min_delay)
#                         current_result = self.generate_with_gemini(initial_prompt)
#                         current_floor_plan = extract_json_from_text(current_result)
#                         if not current_floor_plan:
#                             break
#                 else:
#                     # Edge case: if we didn't find any problems but didn't process all rooms
#                     if min_delay > 0:
#                         time.sleep(min_delay)
#                     current_result = self.generate_with_gemini(initial_prompt)
#                     current_floor_plan = extract_json_from_text(current_result)
#                     if not current_floor_plan:
#                         break
                    
#                 pbar.update(1)
        
#         # Final result
#         final_json = json.dumps(valid_floor_plan, indent=2)
        
#         # Check for any issues in the final floor plan
#         _, final_overlaps, final_invalid_geometries = check_for_overlaps(valid_floor_plan)
#         self.stats["final_overlap_count"] = final_overlaps
#         self.stats["final_invalid_geometry_count"] = final_invalid_geometries
        
#         return final_json, self.stats


import os
import time
import json
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import tqdm
from vllm import LLM, SamplingParams

# (Assume count_tokens, extract_json_from_text, check_for_overlaps are imported)

class IncrementalFloorplanGenerator:
    """
    An incremental generator that verifies rooms as they are added and rewinds when overlaps are detected,
    now backed by a small Llama 4 model via vLLM.
    """
    
    def __init__(
        self,
        model_path: str = "models/Llama-4-Scout-17B-16E-Instruct",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        max_iterations: int = 10
    ):
        """
        Initialize the incremental floor plan generator using vLLM.
        
        Args:
            model_path: Path or identifier of the Llama 4 small model
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            max_tokens: Maximum tokens to generate
            max_iterations: Maximum number of incremental generations
        """
        # Load vLLM model
        self.llm = LLM(model=model_path)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # Generation limits
        self.max_iterations = max_iterations
        
        # Statistics tracking
        self.stats = {
            "llm_call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "iterations_count": 0,
            "rewind_count": 0,
            "final_overlap_count": 0,
            "final_invalid_geometry_count": 0
        }
        
        # Define system prompt for floor plan generation
        self.SYSTEM_PROMPT = """
You are a state-of-the-art floor-plan generator that translates JSON specifications and connectivity requirements defined by a bubble diagram into precise, optimized layouts.
Your algorithm considers each room's dimensions, proportion, and desired adjacencies to produce an efficient arrangement that maximizes usable space while honoring all constraints.
Your top priority is that no two room polygons ever overlap. Rooms must be strictly disjoint, room interiors must never intersect.

Your output must be a JSON object, where `output` key contains:
- `room_count`: the total number of room entries
- `rooms`: a list of rooms. Each room entry in `rooms` must include:
  - `id`: formatted as `<room_type>|<unique_index>` (e.g. `"bedroom|2"` or `"bathroom|0"`)
  - `room_type`: the room type (e.g. `"living_room"`, `"kitchen"`, etc.)
  - `area`: in square meters (all positive numbers)
  - `floor_polygon`: an ordered list of `{x: , y:}` vertices defining a simple polygon

Additional rules:
- **Absolute non-overlap**: no two room polygons may share any interior point under any circumstances.
- Every `id` used in the bubble diagram must appear in the `rooms` list.

Return only a JSON object containing an `output` key without extra commentary or explanation.
"""

    def generate_with_llm(self, prompt: str) -> str:
        """
        Generate a response using vLLM.
        """
        # Count input tokens
        self.stats["input_tokens"] += count_tokens(prompt)

        # Fire off a single vLLM call
        self.stats["llm_call_count"] += 1
        try:
            responses = self.llm.generate([prompt], sampling_params=self.sampling_params)
            # grab the first (and only) response
            resp = next(responses)
            output_text = resp.outputs[0].text
        except Exception:
            return ""
        
        # Count output tokens
        out_toks = count_tokens(output_text)
        self.stats["output_tokens"] += out_toks
        self.stats["total_tokens"] += out_toks
        
        return output_text

    def generate_floor_plan(self, input_data: str, min_delay: int = 10):
        """
        Generate a floor plan with intelligent rewind on overlap detection using vLLM.
        
        Args:
            input_data: Bubble diagram input
            min_delay: Minimum delay between API calls in seconds
            
        Returns:
            Tuple[str, Dict[str, Any]]: (floor plan JSON, stats)
        """
        # Reset stats
        for k in self.stats:
            self.stats[k] = 0
        
        # Build initial prompt
        initial_prompt = f"{self.SYSTEM_PROMPT}\n\nHere is the bubble diagram for the floor plan:\n\n{input_data}"
        
        # First generation
        current_result = self.generate_with_llm(initial_prompt)
        current_floor_plan = extract_json_from_text(current_result)
        if not current_floor_plan:
            return current_result, self.stats
        
        iterations = 0
        valid_floor_plan = {"output": {"room_count": 0, "rooms": []}}
        valid_polygons = []
        valid_room_ids = []
        
        with tqdm(desc="Incremental generation", total=self.max_iterations, disable=True) as pbar:
            while iterations < self.max_iterations:
                iterations += 1
                self.stats["iterations_count"] = iterations
                
                # Extract rooms
                rooms = current_floor_plan.get("output", {}).get("rooms", [])
                
                last_valid_idx = -1
                should_rewind = False
                
                for idx, room in enumerate(rooms):
                    if idx < len(valid_floor_plan["output"]["rooms"]):
                        last_valid_idx = idx
                        continue
                    
                    room_id = room.get("id", f"unknown|{idx}")
                    if room_id in valid_room_ids:
                        last_valid_idx = idx
                        continue
                    
                    # Validate polygon
                    pts = room.get("floor_polygon", [])
                    if len(pts) < 3:
                        should_rewind = True
                        break
                    
                    coords = [(p["x"], p.get("z", p.get("y"))) for p in pts]
                    poly = Polygon(coords)
                    if not poly.is_valid:
                        should_rewind = True
                        break
                    
                    # Overlap check
                    for vp in valid_polygons:
                        if poly.intersects(vp) and not poly.touches(vp):
                            should_rewind = True
                            break
                    if should_rewind:
                        break
                    
                    # Accept room
                    valid_floor_plan["output"]["rooms"].append(room)
                    valid_polygons.append(poly)
                    valid_room_ids.append(room_id)
                    last_valid_idx = idx
                
                valid_floor_plan["output"]["room_count"] = len(valid_floor_plan["output"]["rooms"])
                
                # Done if no rewind needed and all rooms processed
                if not should_rewind and last_valid_idx >= len(rooms) - 1:
                    break
                
                # Rewind logic
                if should_rewind:
                    self.stats["rewind_count"] += 1
                    partial_json = json.dumps(valid_floor_plan, indent=2)
                    rewind_prompt = (
                        f"{self.SYSTEM_PROMPT}\n\nHere is the bubble diagram:\n\n{input_data}\n\n"
                        f"{partial_json}"
                    )
                    if min_delay > 0:
                        time.sleep(min_delay)
                    cont = self.generate_with_llm(rewind_prompt)
                    new_floor_plan = extract_json_from_text(cont)
                    current_floor_plan = new_floor_plan or current_floor_plan
                else:
                    # Unexpected, just retry initial
                    if min_delay > 0:
                        time.sleep(min_delay)
                    initial = self.generate_with_llm(initial_prompt)
                    current_floor_plan = extract_json_from_text(initial) or current_floor_plan
                
                pbar.update(1)
        
        # Finalize
        final_json = json.dumps(valid_floor_plan, indent=2)
        _, overlaps, invalids = check_for_overlaps(valid_floor_plan)
        self.stats["final_overlap_count"] = overlaps
        self.stats["final_invalid_geometry_count"] = invalids
        
        return final_json, self.stats

def check_for_overlaps(floor_plan: Dict) -> Tuple[bool, int, int]:
    """
    Check if any rooms in the floor plan overlap or have invalid geometries.
    
    Args:
        floor_plan: Parsed floor plan JSON
        
    Returns:
        Tuple of (has_problems: bool, overlap_count: int, invalid_geometry_count: int)
    """
    # Make sure we have a valid floor plan with rooms
    if not floor_plan or "output" not in floor_plan or "rooms" not in floor_plan["output"]:
        return True, 0, 0  # Problem: missing structure
        
    rooms = floor_plan["output"]["rooms"]
    if not rooms:
        return True, 0, 0  # Problem: no rooms
        
    # Extract polygons from rooms
    polygons = []
    room_ids = []
    invalid_geometry_count = 0
    
    for room in rooms:
        if "floor_polygon" not in room:
            invalid_geometry_count += 1
            continue
            
        floor_polygon = room["floor_polygon"]
        
        # Check if we have coordinates in the format we expect
        if not floor_polygon or len(floor_polygon) < 3:
            invalid_geometry_count += 1
            continue
            
        # Extract coordinates, handling both y and z formats
        try:
            if "z" in floor_polygon[0]:
                coords = [(p['x'], p['z']) for p in floor_polygon]
            else:
                coords = [(p['x'], p['y']) for p in floor_polygon]
                
            # Create a polygon and add to list
            poly = Polygon(coords)
            if not poly.is_valid:
                invalid_geometry_count += 1
                continue
            
            polygons.append(poly)
            room_ids.append(room.get("id", "unknown"))
        except (KeyError, ValueError):
            invalid_geometry_count += 1
            continue
    
    # Check for overlaps between polygons
    overlap_count = 0
    for i in range(len(polygons)):
        for j in range(i+1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                overlap_count += 1
    
    has_problems = (overlap_count > 0 or invalid_geometry_count > 0)
    return has_problems, overlap_count, invalid_geometry_count


def test_generators(api_key=None, model="gemini-1.5-pro", min_delay=10, max_iterations=10, max_retries=5, temperature=0.7, num_samples=5, split="test"):
    """
    Test all three generators on a subset of samples and compare their performance.
    
    Args:
        api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        model: Gemini model name to use
        min_delay: Minimum delay between API calls in seconds
        max_iterations: Maximum number of iterations for incremental generator
        max_retries: Maximum number of retries for retry generator
        temperature: Sampling temperature
        num_samples: Number of samples to test
        split: Dataset split to use ('train', 'test', or 'validation')
    """
    # Load data from the rplan dataset
    import datasets
    import random
    import os
    from collections import defaultdict
    
    print(f"Loading rplan_converted_no_doors dataset, split: {split}...")
    dataset = datasets.load_dataset("oops-all-pals/rplan_converted_no_doors")
    
    # Create output directory for results
    os.makedirs("results", exist_ok=True)
    
    # Initialize generators
    print("Initializing generators...")
    simple_gen = SimpleFloorplanGenerator(
        api_key=api_key,
        model=model,
        temperature=temperature
    )
    
    retry_gen = RetryFloorplanGenerator(
        api_key=api_key,
        model=model,
        max_retries=max_retries,
        temperature=temperature
    )
    
    incremental_gen = IncrementalFloorplanGenerator(
        api_key=api_key,
        model=model,
        max_iterations=max_iterations,
        temperature=temperature
    )
    
    # Statistics aggregation
    simple_stats_agg = defaultdict(list)
    retry_stats_agg = defaultdict(list)
    incremental_stats_agg = defaultdict(list)
    
    # Process a subset of samples
    subset_size = min(num_samples, len(dataset[split]))
    indices = random.sample(range(len(dataset[split])), subset_size)
    
    print(f"Testing on {subset_size} random samples from the {split} split")
    
    for i, sample_idx in enumerate(indices):
        # Get sample data
        sample_data = dataset[split][sample_idx]
        sample_input = json.dumps(sample_data['input'], indent=2)
        
        print(f"\n--- Sample {i+1}/{subset_size} (Index {sample_idx}) ---")
        print(f"Room count: {sample_data['room_count']}, Total area: {sample_data['total_area']:.1f}")
        
        # Save the original input for reference
        with open(f"results/input_{sample_idx}.json", "w") as f:
            f.write(sample_input)
        
        # Generate with simple approach
        print(f"\nGenerating with simple approach (sample {i+1}/{subset_size})...")
        simple_result, simple_stats = simple_gen.generate_floor_plan(sample_input)
        
        # Add mandatory delay between methods to respect API rate limits
        if min_delay > 0:
            print(f"Waiting {min_delay} seconds between methods...")
            time.sleep(min_delay)
        
        # Generate with retry approach
        print(f"\nGenerating with retry approach (sample {i+1}/{subset_size})...")
        retry_result, retry_stats = retry_gen.generate_floor_plan(
            sample_input, 
            min_delay=min_delay
        )
        
        # Add mandatory delay between methods to respect API rate limits
        if min_delay > 0:
            print(f"Waiting {min_delay} seconds between methods...")
            time.sleep(min_delay)
        
        # Generate with incremental approach
        print(f"\nGenerating with incremental approach (sample {i+1}/{subset_size})...")
        incremental_result, incremental_stats = incremental_gen.generate_floor_plan(
            sample_input, 
            min_delay=min_delay
        )
        
        # Extract JSON for analysis
        simple_extracted = extract_json_from_text(simple_result)
        retry_extracted = extract_json_from_text(retry_result)
        incremental_extracted = extract_json_from_text(incremental_result)
        
        # Save results for this sample
        with open(f"results/simple_{sample_idx}.json", "w") as f:
            f.write(simple_result)
        
        with open(f"results/retry_{sample_idx}.json", "w") as f:
            f.write(retry_result)
        
        with open(f"results/incremental_{sample_idx}.json", "w") as f:
            f.write(incremental_result)
        
        # Aggregate statistics
        for key, value in simple_stats.items():
            simple_stats_agg[key].append(value)
            
        for key, value in retry_stats.items():
            retry_stats_agg[key].append(value)
            
        for key, value in incremental_stats.items():
            incremental_stats_agg[key].append(value)
        
        # Print per-sample results
        print(f"\n=== Results for Sample {i+1}/{subset_size} (Index {sample_idx}) ===")
        print("Simple approach:")
        print(f"  - LLM calls: {simple_stats['llm_call_count']}")
        print(f"  - Total tokens: {simple_stats['total_tokens']}")
        print(f"  - Overlaps: {simple_stats['overlap_count']}")
        print(f"  - Invalid geometries: {simple_stats['invalid_geometry_count']}")
        
        print("\nRetry approach:")
        print(f"  - LLM calls: {retry_stats['llm_call_count']}")
        print(f"  - Total tokens: {retry_stats['total_tokens']}")
        print(f"  - Retry count: {retry_stats['retry_count']}")
        print(f"  - Final overlap count: {retry_stats['final_overlap_count']}")
        print(f"  - Final invalid geometries: {retry_stats['final_invalid_geometry_count']}")
        
        print("\nIncremental approach:")
        print(f"  - LLM calls: {incremental_stats['llm_call_count']}")
        print(f"  - Total tokens: {incremental_stats['total_tokens']}")
        print(f"  - Iterations: {incremental_stats['iterations_count']}")
        print(f"  - Rewind count: {incremental_stats['rewind_count']}")
        print(f"  - Final overlap count: {incremental_stats['final_overlap_count']}")
        print(f"  - Final invalid geometries: {incremental_stats['final_invalid_geometry_count']}")
        
        # Add delay between samples if specified
        if i < subset_size - 1 and min_delay > 0:
            print(f"\nWaiting {min_delay} seconds before next sample...")
            time.sleep(min_delay)
    
    # Calculate averages
    simple_avg = {k: sum(v) / len(v) if v else 0 for k, v in simple_stats_agg.items()}
    retry_avg = {k: sum(v) / len(v) if v else 0 for k, v in retry_stats_agg.items()}
    incremental_avg = {k: sum(v) / len(v) if v else 0 for k, v in incremental_stats_agg.items()}
    
    # Print aggregate statistics
    print("\n" + "="*50)
    print(f"AGGREGATE STATISTICS OVER {subset_size} SAMPLES")
    print("="*50)
    
    print("\nSimple approach averages:")
    print(f"  - Avg. LLM calls: {simple_avg['llm_call_count']:.2f}")
    print(f"  - Avg. total tokens: {simple_avg['total_tokens']:.2f}")
    print(f"  - Avg. overlaps: {simple_avg['overlap_count']:.2f}")
    print(f"  - Avg. invalid geometries: {simple_avg['invalid_geometry_count']:.2f}")
    
    print("\nRetry approach averages:")
    print(f"  - Avg. LLM calls: {retry_avg['llm_call_count']:.2f}")
    print(f"  - Avg. total tokens: {retry_avg['total_tokens']:.2f}")
    print(f"  - Avg. retry count: {retry_avg['retry_count']:.2f}")
    print(f"  - Avg. final overlap count: {retry_avg['final_overlap_count']:.2f}")
    print(f"  - Avg. final invalid geometries: {retry_avg['final_invalid_geometry_count']:.2f}")
    
    print("\nIncremental approach averages:")
    print(f"  - Avg. LLM calls: {incremental_avg['llm_call_count']:.2f}")
    print(f"  - Avg. total tokens: {incremental_avg['total_tokens']:.2f}")
    print(f"  - Avg. iterations: {incremental_avg['iterations_count']:.2f}")
    print(f"  - Avg. rewind count: {incremental_avg['rewind_count']:.2f}")
    print(f"  - Avg. final overlap count: {incremental_avg['final_overlap_count']:.2f}")
    print(f"  - Avg. final invalid geometries: {incremental_avg['final_invalid_geometry_count']:.2f}")
    
    # Calculate token savings
    retry_token_savings = simple_avg['total_tokens'] - retry_avg['total_tokens']
    retry_savings_pct = (retry_token_savings / simple_avg['total_tokens']) * 100 if simple_avg['total_tokens'] > 0 else 0
    
    incremental_token_savings = simple_avg['total_tokens'] - incremental_avg['total_tokens'] 
    incremental_savings_pct = (incremental_token_savings / simple_avg['total_tokens']) * 100 if simple_avg['total_tokens'] > 0 else 0
    
    incremental_vs_retry_savings = retry_avg['total_tokens'] - incremental_avg['total_tokens']
    incremental_vs_retry_pct = (incremental_vs_retry_savings / retry_avg['total_tokens']) * 100 if retry_avg['total_tokens'] > 0 else 0
    
    print("\nToken efficiency:")
    print(f"  - Retry vs Simple: {retry_token_savings:.2f} tokens saved ({retry_savings_pct:.1f}%)")
    print(f"  - Incremental vs Simple: {incremental_token_savings:.2f} tokens saved ({incremental_savings_pct:.1f}%)")
    print(f"  - Incremental vs Retry: {incremental_vs_retry_savings:.2f} tokens saved ({incremental_vs_retry_pct:.1f}%)")
    
    # Calculate average problem reduction
    simple_problems = simple_avg['overlap_count'] + simple_avg['invalid_geometry_count']
    retry_problems = retry_avg['final_overlap_count'] + retry_avg['final_invalid_geometry_count']
    incr_problems = incremental_avg['final_overlap_count'] + incremental_avg['final_invalid_geometry_count']
    
    if simple_problems > 0:
        retry_reduction = simple_problems - retry_problems
        retry_reduction_pct = (retry_reduction / simple_problems) * 100 if simple_problems > 0 else 0
        
        incr_reduction = simple_problems - incr_problems
        incr_reduction_pct = (incr_reduction / simple_problems) * 100 if simple_problems > 0 else 0
        
        print("\nProblem reduction:")
        print(f"  - Retry vs Simple: {retry_reduction:.2f} problems reduced ({retry_reduction_pct:.1f}%)")
        print(f"  - Incremental vs Simple: {incr_reduction:.2f} problems reduced ({incr_reduction_pct:.1f}%)")
    
    # Save aggregate statistics to file
    aggregate_results = {
        "num_samples": subset_size,
        "simple_averages": simple_avg,
        "retry_averages": retry_avg,
        "incremental_averages": incremental_avg,
        "token_savings": {
            "retry_vs_simple": {
                "absolute": retry_token_savings,
                "percentage": retry_savings_pct
            },
            "incremental_vs_simple": {
                "absolute": incremental_token_savings,
                "percentage": incremental_savings_pct
            },
            "incremental_vs_retry": {
                "absolute": incremental_vs_retry_savings,
                "percentage": incremental_vs_retry_pct
            }
        },
        "problem_reduction": {
            "retry_vs_simple": {
                "absolute": retry_reduction if simple_problems > 0 else 0,
                "percentage": retry_reduction_pct if simple_problems > 0 else 0
            },
            "incremental_vs_simple": {
                "absolute": incr_reduction if simple_problems > 0 else 0,
                "percentage": incr_reduction_pct if simple_problems > 0 else 0
            }
        }
    }
    
    with open("results/aggregate_statistics.json", "w") as f:
        json.dump(aggregate_results, f, indent=2)
        
    print("\nResults saved to 'results/' directory")
    print("Aggregate statistics saved to 'results/aggregate_statistics.json'")


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test floor plan generation approaches")
    parser.add_argument("--api_key", type=str, default=None, help="Google API key (defaults to GOOGLE_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro", help="Gemini model to use")
    parser.add_argument("--delay", type=int, default=10, help="Minimum delay between API calls in seconds")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum number of iterations for incremental generator")
    parser.add_argument("--retries", type=int, default=5, help="Maximum number of retries for retry generator")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for dataset sampling")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "validation"], help="Dataset split to use")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        import random
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Run the test with the specified parameters
    test_generators(
        api_key=args.api_key,
        model=args.model,
        min_delay=args.delay,
        max_iterations=args.iterations,
        max_retries=args.retries,
        temperature=args.temperature,
        num_samples=args.samples,
        split=args.split
    )