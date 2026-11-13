import warnings

from typing import Sequence

import numpy as np


class JointOrderingConverter:
    """
    Converts joint quantities from a source training environment to a target environment.
    """

    def __init__(
        self,
        training_sim_joint_order: Sequence[str],
        target_sim_joint_order: Sequence[str],
    ) -> None:
        """Convert a joint ordering Sequence into another one.

        Args:
          training_sim_joint_order: List specifying the joint order in the source simulator where the policy was trained
          target_sim_joint_order: List specifying the joint order in the target environment where the policy is deployed.
        """

        training_sim_joint_ordering_tmp: list[int] = []
        for joint_name in training_sim_joint_order:
            if joint_name in target_sim_joint_order:
                training_sim_joint_ordering_tmp.append(
                    target_sim_joint_order.index(joint_name)
                )
            else:
                warnings.warn(
                    f'Could not find joint "{joint_name}" in the target joint order'
                )

        self._training_sim_joint_ordering = np.array(training_sim_joint_ordering_tmp)
        """Indexing vector to transform a vector of the source joint ordering into the target joint ordering"""

        self._target_sim_joint_ordering = np.empty_like(
            self._training_sim_joint_ordering
        )
        """Indexing vector to transform a vector in the target joint ordering into the source joint ordering"""

        for isaac_i, pr_j in enumerate(self._training_sim_joint_ordering):
            self._target_sim_joint_ordering[pr_j] = isaac_i

    def to_source(self, vec: Sequence[float]) -> Sequence[float]:
        """Convert a vector from target ordering -> training source ordering.

        Args:
            vec: The vector of joint quantities to be transformed

        Returns:
            The transformed vector
        """
        return vec[self._training_sim_joint_ordering]

    def from_source(self, vec: Sequence[float]) -> Sequence[float]:
        """Convert a vector from training source ordering -> target ordering.

        Args:
            vec: The vector of joint quantities to be transformed

        Returns:
            The transformed vector
        """
        return vec[self._target_sim_joint_ordering]
