//! AllyH compact Agent-Computer Interface (ACI) tools for `rig-core`.

pub mod command;
pub mod edit;
pub mod error;
pub mod git;
pub mod inspect;
pub mod mcp_client;
pub mod registry;
pub mod search;
pub mod sim;
pub mod skill;
pub mod verify;

pub use command::{
    CommandPolicyDecision, CommandPolicyGate, PendingCommandApproval, RunCommandArgs,
    RunCommandOutput, RunCommandTool, suggest_pattern,
};
pub use edit::{EditByHashArgs, EditByHashOutput, EditByHashTool, EditHunk};
pub use error::{Result, ToolError};
pub use git::{GitInspectArgs, GitInspectOutput, GitInspectTool};
pub use inspect::{
    InspectSymbolArgs, InspectSymbolOutput, InspectSymbolTool, InspectedFileSnippet,
};
pub use mcp_client::{
    MCP_PROTOCOL_VERSION, McpClientManager, McpRemoteTool, McpServerStatus, parse_mcp_server_spec,
    parse_sse_json_rpc,
};
pub use registry::{AciToolset, ToolExecutionRecord};
pub use search::{SearchCodeArgs, SearchCodeOutput, SearchCodeTool, SemanticCodeHit};
pub use sim::{InspectSimArgs, InspectSimOutput, InspectSimTool};
pub use skill::{
    AddSkillArgs, AddSkillOutput, AddSkillTool, GetSkillArgs, GetSkillOutput, GetSkillTool,
};
pub use verify::{RunVerificationArgs, RunVerificationOutput, RunVerificationTool};
