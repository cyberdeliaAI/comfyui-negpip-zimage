from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .src.nodes import NegPipPrompt


class NegPipPromptExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [NegPipPrompt]


async def comfy_entrypoint():
    return NegPipPromptExtension()
