"""Variable schema definitions for disclosure-risk assessments."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["qi", "sensitive", "stratum", "id", "ignore"]
VarType = Literal["categorical", "ordinal", "numeric", "date"]


class Variable(BaseModel):
    """Describe one column and its analytic roles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    vtype: VarType
    role: list[Role] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    tolerance: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def tolerance_requires_numeric_type(self) -> "Variable":
        if self.tolerance is not None and self.vtype != "numeric":
            raise ValueError("tolerance is supported only for numeric variables")
        return self


class Schema(BaseModel):
    """Define variables available to risk metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variables: list[Variable] = Field(min_length=1)

    @model_validator(mode="after")
    def variable_names_are_unique(self) -> "Schema":
        names = [variable.name for variable in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("variable names must be unique")
        return self

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Return the schema's variable names in declaration order."""
        return tuple(variable.name for variable in self.variables)

    def variables_with_role(self, role: Role) -> tuple[Variable, ...]:
        """Return variables that declare the requested role."""
        return tuple(variable for variable in self.variables if role in variable.role)
