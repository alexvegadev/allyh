//! AllyH compact Agent-Computer Interface (ACI) tools for `rig-core`.

pub mod edit;
pub mod error;
pub mod inspect;
pub mod registry;
pub mod sim;
pub mod skill;
pub mod verify;

pub use edit::{EditByHashArgs, EditByHashOutput, EditByHashTool};
pub use error::{Result, ToolError};
pub use inspect::{
    InspectSymbolArgs, InspectSymbolOutput, InspectSymbolTool, InspectedFileSnippet,
};
pub use registry::{AciToolset, ToolExecutionRecord};
pub use sim::{InspectSimArgs, InspectSimOutput, InspectSimTool};
pub use skill::{
    AddSkillArgs, AddSkillOutput, AddSkillTool, GetSkillArgs, GetSkillOutput, GetSkillTool,
};
pub use verify::{RunVerificationArgs, RunVerificationOutput, RunVerificationTool};
