# MPI
 BTP-I-MPI-Renderer

# Instructions
Use test_file.py for viewing non re-factored code.

# Requirements

tensorflow==2.2.0
tensorflow-addons==0.10.0
_(No need for GPU version)_

# Re-factoring
break monolith into:

**render.py**: driver file

**functions.py**: matrix and helper functions

**graphics.py**: plane, surface generation/removal, other graphics functions

**homography.py**: warper & inverse homographer
