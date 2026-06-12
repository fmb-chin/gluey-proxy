def _convert_function_call_to_namespaced(
    output_items: list, mcp_namespace_map: dict
) -> list:
    """Rewrite function_call items that match flattened MCP tools so the Codex
    client can route them to the correct MCP server.

    When we flatten a namespace like mcp__firecrawl -> mcp__firecrawl__firecrawl_search,
    the Ollama model generates a function_call with name="mcp__firecrawl__firecrawl_search"
    and namespace=None. The Codex client's tool router handles ResponseItem::FunctionCall
    by constructing ToolName::new(namespace, name). If the namespace is present, it
    creates a namespaced ToolName that matches the registered MCP handler. If absent,
    it creates a plain ToolName that won't match.

    IMPORTANT: We must keep type="function_call" because CustomToolCall uses
    ToolName::plain(name), which ignores the namespace field entirely.
    """
    if not mcp_namespace_map:
        return output_items
    converted = []
    for item in output_items:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        if item.get("type") != "function_call":
            converted.append(item)
            continue
        name = item.get("name", "")
        mapping = mcp_namespace_map.get(name)
        if not mapping:
            converted.append(item)
            continue
        new_item = dict(item)
        new_item["name"] = mapping["name"]
        new_item["namespace"] = mapping["namespace"]
        converted.append(new_item)
    return converted


def _convert_mcp_calls_in_response(
    resp: dict, mcp_namespace_map: dict, request_annotations: dict
) -> dict:
    """Walk a full Responses API dict and rewrite flattened MCP function calls."""
    if not mcp_namespace_map:
        return resp
    output = resp.get("output", [])
    new_output = _convert_function_call_to_namespaced(output, mcp_namespace_map)
    changed = len(new_output) != len(output) or any(
        a is not b for a, b in zip(new_output, output)
    )
    if changed:
        resp = dict(resp)
        resp["output"] = new_output
        request_annotations.setdefault("mcp_call_converted", 0)
        request_annotations["mcp_call_converted"] += 1
    return resp
