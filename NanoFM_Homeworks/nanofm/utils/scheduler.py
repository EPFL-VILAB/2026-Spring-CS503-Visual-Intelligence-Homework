# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on DINO code base
# https://github.com/facebookresearch/dino
# --------------------------------------------------------

import numpy as np
import math

def cosine_scheduler(base_value, final_value, total_iters, warmup_iters, start_warmup_value=0.0):
    assert warmup_iters >= 0
    assert total_iters > 0
    assert start_warmup_value <= base_value
    assert base_value >= final_value

    if warmup_iters > 0:
        print("Set warmup iters = %d" % warmup_iters)
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    else:
        print("No warmup iters")
        warmup_schedule = np.array([])

    cosine_iters = np.arange(total_iters - warmup_iters)
    cosine_schedule = np.array([
        final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(cosine_iters)))) 
        for i in cosine_iters
    ])

    schedule = np.concatenate((warmup_schedule, cosine_schedule))

    assert len(schedule) == total_iters
    return schedule
