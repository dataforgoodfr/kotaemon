from typing import AsyncGenerator, Iterator

from kotaemon.base import BaseComponent, Document
<<<<<<< HEAD
=======
from pydantic import BaseModel
>>>>>>> 97a2958 ((feat) kotaemon : add libs taxonomy OK & pipelineblocks OK)

class BasePdfExtractionBlock(BaseComponent):

    """ A simple base class for pdf extraction blocks"""

    def stream(self, *args, **kwargs) -> Iterator[Document] | None :
        raise NotImplementedError

    def astream(self, *args, **kwargs) -> AsyncGenerator[Document, None] | None:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> Document | list[Document] | Iterator[Document] | None:
        return NotImplementedError