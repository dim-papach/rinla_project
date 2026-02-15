> can i add new methods other than inla in my cli?

✦ Yes, absolutely. Currently, your CLI is tightly coupled to
  the INLA method, but we can refactor it to support multiple
  algorithms (e.g., mean, median, scikit-image inpainting, or
  other statistical methods) using a Strategy Pattern.

  Here is the high-level plan to achieve this:

   1. Create a Processor Interface: Define an abstract base
      class (e.g., ImageRestorer) that all methods must
      implement. This ensures every method has a consistent
      .process() function.
   2. Refactor INLA Logic: Move the current INLA-specific code
      from FitsProcessor into a dedicated INLAProcessor class.
   3. Add New Methods: Implement other classes like
      MeanProcessor or MedianProcessor that implement the
      interface.
   4. Update CLI: Add a --method argument to the process
      command to let the user select the algorithm.

  Proposed Architecture

  `fyf/core/processing/base.py`

   1 from abc import ABC, abstractmethod
   2 import numpy as np
   3 
   4 class BaseProcessor(ABC):
   5     @abstractmethod
   6     def process(self, data: np.ndarray, mask: np.ndarray,
     config: dict) -> np.ndarray:
   7         pass

  `fyf/core/processing/simple.py`

   1 class MeanProcessor(BaseProcessor):
   2     def process(self, data: np.ndarray, mask: np.ndarray,
     config: dict) -> np.ndarray:
   3         # Simple implementation: fill masked areas with 
     local mean
   4         return filled_data

  Would you like me to refactor the code to support this
  "plug-and-play" architecture now? This would allow you to
  easily add new methods in the future.

