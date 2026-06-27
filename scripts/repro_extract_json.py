"""Reproduce extract_json_object failure with actual error outputs."""
import json
from src.generation.endpoint import EndpointLLMClient


def main():
    with open("inference_errors.json", encoding="utf-8") as f:
        errors = json.load(f)

    for entry in errors:
        error_msg = entry["error"]
        # Extract the output part after the prefix
        prefix = "OpenRouter trả về JSON không hợp lệ sau 3 lần thử:\n"
        if not error_msg.startswith(prefix):
            print(f"[SKIP id={entry['id']}] Unknown error format")
            continue
        output = error_msg[len(prefix):]
        print(f"\n{'='*60}")
        print(f"id={entry['id']}: {entry['question'][:60]}...")
        print(f"Output length: {len(output)} chars")
        print(f"First 100 chars: {repr(output[:100])}")
        print(f"Has BOM: {output.startswith(chr(0xFEFF))}")
        print(f"Starts with '{{': {output.startswith('{')}")
        
        # Try extract_json_object
        try:
            result = EndpointLLMClient.extract_json_object(output)
            print(f"✓ PARSED OK: {list(result.keys())}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"✗ FAILED: {e}")
            # Debug: try raw json.loads
            try:
                json.loads(output)
                print("  (but json.loads on raw output works)")
            except json.JSONDecodeError as inner:
                print(f"  json.loads also fails: {inner}")
            
            # Debug: try brace-counting manually
            start = output.find("{")
            depth = 0
            in_string = False
            escape = False
            for end in range(start, len(output)):
                ch = output[end]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                if depth == 0:
                    candidate = output[start : end + 1]
                    print(f"  Found candidate at [{start}:{end+1}]")
                    try:
                        json.loads(candidate)
                        print(f"  ✓ candidate is valid JSON")
                    except json.JSONDecodeError as e2:
                        print(f"  ✗ candidate not valid: {e2}")
                    break
            else:
                print(f"  No depth-0 found in entire output")
            
            if depth != 0:
                print(f"  Final depth={depth}, never closed")


if __name__ == "__main__":
    main()