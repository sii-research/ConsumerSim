from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .client import ConsumerSimBackend


mcp = FastMCP(
    "consumersim",
    instructions=(
        "ConsumerSim public MCP proxy. Tools accept region and time inputs, "
        "then forward requests to the private ConsumerSim backend."
    ),
)


@mcp.tool()
async def forecast_lookup(region: str, month: str, week: str | int | None = None) -> dict[str, Any]:
    return await ConsumerSimBackend().forecast_lookup(region=region, month=month, week=week)


@mcp.tool()
async def forecast_times(region: str | None = None) -> dict[str, Any]:
    return await ConsumerSimBackend().forecast_times(region=region)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
