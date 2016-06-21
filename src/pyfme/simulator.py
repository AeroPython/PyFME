"""
Python Flight Mechanics Engine (PyFME).
Copyright (c) AeroPython Development Team.
Distributed under the terms of the MIT License.

Simulation class
----------------

"""
from abc import abstractmethod


class Simulation(object):

    def __init__(self, aircraft, system, environment):
        self.aircraft = aircraft
        self.system = system
        self.environment = environment
        self._time_step = 0

    def time_step(self, dt):

        self._time_step += 1
        self.system.propagate(self.aircraft, dt)
        self.environment.update(self.system)
        controls = self._get_current_controls(self._time_step)
        self.aircraft.update(controls, self.system, self.environment)
        self.aircraft.calculate_forces_and_moments()

    @abstractmethod
    def _get_current_controls(self, ii):
        return


class BatchSimulation(Simulation):

    def __init__(self, aircraft, system, environment):
        Simulation.__init__(self, aircraft, system, environment)
        self.time = None
        self.aircraft_controls = {}

    def set_controls(self, time, controls):
        """Set the time history of controls and the corresponding times.

        Parameters
        ----------
        time
        controls

        Returns
        -------

        """

        # check time dimensions
        if time.ndim != 1:
            raise ValueError('Time must be unidimensional')
        tsize = time.size
        # check dimensions
        for c in controls:
            if controls[c].size != tsize:
                msg = 'Control {} size ({}) does not match time size ({' \
                      '})'.fromat(c, c.size, tsize)
                raise ValueError(msg)
        self.time = time
        self.aircraft_controls = controls

    def _get_current_controls(self, ii):
        """Returns controls at time step ii.

        Parameters
        ----------
        ii

        Returns
        -------

        """
        return {control: self.aircraft_controls[control][ii] for control in \
                self.aircraft_controls.keys()}

    def run_simulation(self):

        for ii, t in enumerate(self.time[1:]):
            dt = t - self.time[ii]
            self.time_step(dt)



class RealTimeSimulation(Simulation):

    def __init__(self, aircraft, system, environment):
        super(RealTimeSimulation, self).__init__(aircraft, system, environment)
        # TODO:...

    def _get_current_controls(self, ii):
        # Joystick reading
        raise NotImplementedError
