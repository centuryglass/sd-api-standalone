"""Defines an input parameter and the associated value for a ControlNet preprocessor or unit."""
import json
from typing import Optional, TypeAlias, cast, TypedDict, Any, Callable

from intrapaint_api.util.parameter import Parameter, TYPE_BOOL, TYPE_STR, TYPE_FLOAT, TYPE_INT

ControlParamType: TypeAlias = int | float | str | bool
ControlParamTypeList: TypeAlias = list[int] | list[float] | list[str] | list[bool]

CONTROL_PARAMETER_TYPES = [TYPE_INT, TYPE_FLOAT, TYPE_STR, TYPE_BOOL]


class _ValueSignal:
    """A minimal, Qt-free stand-in for a PySide signal: register callbacks and notify them on emit.

    Replaces the QObject/Signal the class originally used so external consumers can still subscribe to value
    changes without pulling in Qt.
    """

    def __init__(self) -> None:
        self._callbacks: list[Callable[[ControlParamType], None]] = []

    def connect(self, callback: Callable[[ControlParamType], None]) -> None:
        """Register a callback to be invoked with the new value on each change."""
        self._callbacks.append(callback)

    def disconnect(self, callback: Optional[Callable[[ControlParamType], None]] = None) -> None:
        """Remove a previously registered callback, or all callbacks when none is given."""
        if callback is None:
            self._callbacks.clear()
        elif callback in self._callbacks:
            self._callbacks.remove(callback)

    def emit(self, value: ControlParamType) -> None:
        """Notify every connected callback of a new value."""
        for callback in list(self._callbacks):
            callback(value)


class ControlParameter:
    """Defines an input parameter and the associated value for a ControlNet preprocessor or unit."""

    def __init__(self,
                 key: str,
                 display_name: str,
                 value_type: str,
                 default_value: ControlParamType,
                 description: str = '',
                 minimum: Optional[int | float] = None,
                 maximum: Optional[int | float] = None,
                 single_step: Optional[int | float] = None,
                 options: Optional[ControlParamTypeList] = None):
        """
        Initializes a new ControlParameter, setting its value to the default.

        Parameters:
        -----------
        key: str
            The string used to reference the parameter in API requests and responses.
        display_name: str
            The name this parameter uses when labeling associated input widgets
        value_type: str
            A string identifying the type of value being stored.  Valid options are defined in intrapaint_api.util.parameter.py
            and listed in intrapaint_api.api.controlnet.control_parameter.py as CONTROL_PARAMETER_TYPES.
        default_value: ControlParamType
            Initial parameter value, to use if no alternative is specified.
        description: str = ''
            Description string to use as a tooltip on associated control widgets.
        minimum: Optional[int | float] = None
            Minimum permitted value, ignored if the parameter is not an int or float.
        maximum: Optional[int | float] = None
            Maximum permitted value, ignored if the parameter is not an int or float.
        single_step: Optional[int | float] = None
            Minimum interval between accepted values, ignored if the parameter is not an int or float.
        options: Optional[ControlParamTypeList] = None
            If not None, accepted values will be limited to the entries in thie slist.
        """
        if value_type not in CONTROL_PARAMETER_TYPES:
            raise ValueError(f'Invalid ControlNet parameter type for {key}: {value_type}')
        self.value_changed = _ValueSignal()
        self._parameter = Parameter(display_name, value_type, default_value, description, minimum, maximum, single_step)
        self._key = key
        self._value = default_value
        self._multiline = False

        if options is not None:
            self._parameter.set_valid_options(options)

    def __deepcopy__(self, memo: dict[int, Any]) -> 'ControlParameter':
        copy_param = ControlParameter.deserialize(self.serialize())
        memo[id(self)] = copy_param
        return copy_param

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ControlParameter):
            return False
        return self._key == other.key and self.value == other.value \
            and self._multiline == other._multiline and self._parameter == other._parameter

    @property
    def display_name(self) -> str:
        """Returns the parameter's display name."""
        return self._parameter.name

    @property
    def key(self) -> str:
        """Returns the key string used when applying this parameter."""
        return self._key

    @property
    def value(self) -> ControlParamType:
        """Accesses the value currently stored with this parameter."""
        return self._value

    @value.setter
    def value(self, new_value: ControlParamType) -> None:
        if new_value == self._value:
            return
        self._parameter.validate(new_value, True)
        self._value = new_value
        self.value_changed.emit(new_value)

    @property
    def default_value(self) -> ControlParamType:
        """Returns the parameter's default value."""
        default_value = self._parameter.default_value
        assert isinstance(default_value, (int, float, str, bool))
        return default_value

    def set_multiline(self, multiline: bool) -> None:
        """Sets whether this parameter should use a multi-line text box.  This will be ignored if the parameter is not
           a string without specific options."""
        self._multiline = multiline

    class _DataFormat(TypedDict):
        key: str
        value: ControlParamType
        parameter: str
        multiline: bool

    def serialize(self) -> str:
        """Serialize this control parameter to a JSON string."""
        data_dict: ControlParameter._DataFormat = {
            'key': self._key,
            'value': self._value,
            'parameter': self._parameter.serialize(),
            'multiline': self._multiline
        }
        return json.dumps(data_dict)

    @staticmethod
    def deserialize(data_str: str) -> 'ControlParameter':
        """Parse a ControlParameter from serialized text data."""
        data_dict = cast(ControlParameter._DataFormat, json.loads(data_str))
        parameter = Parameter.deserialize(data_dict['parameter'])
        value = data_dict['value']
        default_value = parameter.default_value
        min_value = parameter.minimum
        max_value = parameter.maximum
        options = cast(ControlParamTypeList, parameter.options)
        assert isinstance(default_value, (int, float, str, bool))
        assert min_value is None or isinstance(min_value, (int, float))
        assert max_value is None or isinstance(max_value, (int, float))
        assert options is None or isinstance(options, list)
        control_param = ControlParameter(data_dict['key'], parameter.name, parameter.type_name, default_value,
                                         parameter.description, min_value, max_value, parameter.single_step,
                                         options)
        control_param.value = value
        if data_dict['multiline'] is not None:
            control_param.set_multiline(data_dict['multiline'])
        return control_param
