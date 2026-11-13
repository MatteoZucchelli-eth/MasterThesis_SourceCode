Development
===========

Container workflow
------------------

Start the simulation development stack:

.. code-block:: bash

   make control-sim

The command starts the containers in the background. Attach with VS Code Dev
Containers or:

.. code-block:: bash

   docker compose -f docker/docker-compose.yaml exec dev-container bash

The repository is mounted at /workspace. Build changes inside the container:

.. code-block:: bash

   colcon build
   source install/setup.bash

Use make control for the container without a simulator, or make
teleoperation for the simulator and teleoperation sidecars.

Formatting
----------

Install and run the repository hooks:

.. code-block:: bash

   pipx install pre-commit==4.3.0
   pre-commit install
   pre-commit run --all-files

Documentation
-------------

Build and open the documentation with:

.. code-block:: bash

   make build-docs

C++ debugging
-------------

Build the package with debug symbols:

.. code-block:: bash

   colcon build --packages-select <package> --cmake-args -DCMAKE_BUILD_TYPE=Debug

Launch the node through gdbserver:

.. code-block:: bash

   ros2 run --prefix "gdbserver localhost:3000" <package> <executable>

Then attach from another shell:

.. code-block:: bash

   gdb
   (gdb) target remote localhost:3000
