# MIT License
#
# Copyright (c) 2025 Mobile Robotics Lab, ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Aggregated re-export of all g1 constants.

Definitions live in :mod:`g1_definitions.topics`, :mod:`g1_definitions.joints`,
:mod:`g1_definitions.dataset` and :mod:`g1_definitions.states`; this module
provides the ``from g1_definitions import definitions`` import path.
"""

from g1_definitions.dataset import *  # noqa: F401,F403
from g1_definitions.dataset import _build_slices  # noqa: F401
from g1_definitions.joints import *  # noqa: F401,F403
from g1_definitions.states import *  # noqa: F401,F403
from g1_definitions.topics import *  # noqa: F401,F403
