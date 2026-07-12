from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .src.nodes import ZImageNegPipPrompt


class NegPipZImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ZImageNegPipPrompt]


async def comfy_entrypoint():
    return NegPipZImageExtension()
