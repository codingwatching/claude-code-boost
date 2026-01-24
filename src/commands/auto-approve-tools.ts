import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import type {
  HookOutput,
  ToolDecision,
  PermissionMode,
} from '../types/hook-schemas.js';
import { parseHookInput, ToolDecisionSchema } from '../types/hook-schemas.js';
import { logApproval } from '../logger.js';
import { loadConfig } from '../utils/config.js';
import { getCachedDecision, setCachedDecision } from '../utils/cache.js';
import { log } from '../utils/general-logger.js';
import { getLLMClient, canConfigureLLMClient } from '../utils/llm-client.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function loadSystemPrompt(): string {
  const promptPath = join(__dirname, '../../prompts/system-prompt.md');
  return readFileSync(promptPath, 'utf8');
}

function loadUserPromptTemplate(): string {
  const promptPath = join(__dirname, '../../prompts/user-prompt.md');
  return readFileSync(promptPath, 'utf8');
}

function buildUserPrompt(
  toolName: string,
  toolInput: Record<string, unknown>,
  permissionMode: PermissionMode,
  cwd: string
): string {
  const template = loadUserPromptTemplate();
  return template
    .replace('{{toolName}}', toolName)
    .replace('{{toolInput}}', JSON.stringify(toolInput, null, 2))
    .replace('{{permissionMode}}', permissionMode)
    .replace('{{cwd}}', cwd);
}

function getToolDecisionJsonSchema() {
  return {
    name: 'tool_decision',
    strict: true,
    schema: {
      type: 'object' as const,
      properties: {
        decision: {
          type: 'string' as const,
          enum: ['allow', 'deny', 'ask'],
          description: 'The approval decision for the tool execution',
        },
        reason: {
          type: 'string' as const,
          description: 'Human-readable explanation for the decision',
        },
      },
      required: ['decision', 'reason'],
      additionalProperties: false,
    },
  };
}

async function queryLLM(
  toolName: string,
  toolInput: Record<string, unknown>,
  permissionMode: PermissionMode,
  cwd: string
): Promise<ToolDecision> {
  const llmClient = getLLMClient();
  const systemPrompt = loadSystemPrompt();
  const userPrompt = buildUserPrompt(toolName, toolInput, permissionMode, cwd);

  const response = await llmClient.chatCompletion(
    systemPrompt,
    userPrompt,
    {
      model: llmClient.getModel(),
      maxTokens: 1000,
    },
    getToolDecisionJsonSchema()
  );

  return llmClient.parseJsonResponse<ToolDecision>(
    response.content,
    ToolDecisionSchema
  );
}

// Read-only tools - safe in ALL modes
const READ_ONLY_TOOLS = new Set([
  'Read',
  'LS',
  'LSP',
  'Glob',
  'Grep',
  'WebFetch',
  'WebSearch',
  'NotebookRead',
  'TodoRead',
  'Task',
  'BashOutput',
  'Skill',
  'SlashCommand',
]);

// Interactive tools - always ask (except dontAsk/bypassPermissions)
const INTERACTIVE_TOOLS = new Set([
  'AskUserQuestion',
  'EnterPlanMode',
  'ExitPlanMode',
]);

// Write/mutation tools - behavior depends on mode
const MUTATING_TOOLS = new Set([
  'Write',
  'Edit',
  'MultiEdit',
  'NotebookEdit',
  'TodoWrite',
]);

// Tools that require LLM analysis (e.g., Bash commands)
const LLM_ANALYZED_TOOLS = new Set(['Bash', 'KillShell']);

// Check if a tool is a known Claude Code tool
function isKnownTool(toolName: string): boolean {
  return (
    READ_ONLY_TOOLS.has(toolName) ||
    INTERACTIVE_TOOLS.has(toolName) ||
    MUTATING_TOOLS.has(toolName) ||
    LLM_ANALYZED_TOOLS.has(toolName) ||
    isMcpTool(toolName)
  );
}

// Check if a tool is an MCP tool (prefixed with mcp__)
function isMcpTool(toolName: string): boolean {
  return toolName.startsWith('mcp__');
}

// Apply permission mode adjustments to decision
function applyPermissionModeToDecision(
  decision: 'allow' | 'deny' | 'ask',
  permissionMode: PermissionMode
): 'allow' | 'deny' | 'ask' {
  if (permissionMode === 'dontAsk' && decision === 'ask') {
    return 'allow'; // Permissive in dontAsk mode
  }
  return decision;
}

// Create a HookOutput from a decision
function createHookOutput(
  decision: 'allow' | 'deny' | 'ask',
  reason: string
): HookOutput {
  return {
    hookSpecificOutput: {
      hookEventName: 'PreToolUse',
      permissionDecision: decision,
      permissionDecisionReason: reason,
    },
  };
}

function shouldFastApprove(
  toolName: string,
  toolInput: Record<string, unknown>,
  permissionMode: PermissionMode
): HookOutput | null {
  // bypassPermissions: approve everything immediately
  if (permissionMode === 'bypassPermissions') {
    return createHookOutput(
      'allow',
      `${toolName} auto-approved in bypassPermissions mode`
    );
  }

  // Read-only tools: always allow in all modes
  if (READ_ONLY_TOOLS.has(toolName)) {
    return createHookOutput(
      'allow',
      `${toolName} is a safe read-only operation`
    );
  }

  // Interactive tools: handle based on permission mode
  if (INTERACTIVE_TOOLS.has(toolName)) {
    if (permissionMode === 'dontAsk') {
      return createHookOutput(
        'allow',
        `${toolName} auto-approved in dontAsk mode`
      );
    }

    // For AskUserQuestion, include the question in the reason
    if (toolName === 'AskUserQuestion') {
      const potentialQuestion = toolInput['question'];
      const question =
        typeof potentialQuestion === 'string' ? potentialQuestion : null;
      return createHookOutput(
        'ask',
        question
          ? `Passing question to user: "${question}"`
          : 'AskUserQuestion requires a user response'
      );
    }

    return createHookOutput(
      'ask',
      `${toolName} requires user confirmation before proceeding`
    );
  }

  // Mutating tools: behavior depends on permission mode
  if (MUTATING_TOOLS.has(toolName)) {
    // plan mode: deny all mutations
    if (permissionMode === 'plan') {
      return createHookOutput(
        'deny',
        `${toolName} denied in plan mode - only read operations allowed`
      );
    }

    // acceptEdits mode: allow file edits
    if (permissionMode === 'acceptEdits') {
      return createHookOutput(
        'allow',
        `${toolName} auto-approved in acceptEdits mode`
      );
    }

    // dontAsk mode: allow mutations (permissive)
    if (permissionMode === 'dontAsk') {
      return createHookOutput(
        'allow',
        `${toolName} auto-approved in dontAsk mode`
      );
    }

    // default mode: allow write operations (original CCB behavior)
    return createHookOutput(
      'allow',
      `${toolName} is a safe development operation`
    );
  }

  // MCP tools: handle based on permission mode
  if (isMcpTool(toolName)) {
    // plan mode: allow MCP tools (assume read-only unless LLM says otherwise)
    // dontAsk/acceptEdits/default: allow MCP tools
    return createHookOutput('allow', `${toolName} is an MCP tool`);
  }

  // LLM-analyzed tools (Bash, KillShell): fall through to AI query
  if (LLM_ANALYZED_TOOLS.has(toolName)) {
    return null;
  }

  // Unknown tools: deny with explanation
  if (!isKnownTool(toolName)) {
    return createHookOutput(
      'deny',
      `${toolName} is not a recognized Claude Code tool`
    );
  }

  // Fallback: should not reach here, but fall through to AI query if it does
  return null;
}

export async function autoApproveTools(noCache?: boolean): Promise<void> {
  try {
    const input = readFileSync(0, 'utf8');
    const jsonData = JSON.parse(input);
    const hookData = parseHookInput(jsonData);
    const workingDir = hookData.cwd || process.cwd();
    const permissionMode = hookData.permission_mode;
    const config = loadConfig();

    log.debug(
      {
        tool: hookData.tool_name,
        input: hookData.tool_input,
        sessionId: hookData.session_id,
        cwd: workingDir,
        permissionMode: permissionMode,
        noCache: noCache,
        cacheEnabled: config.cache,
      },
      'Processing tool approval request'
    );

    let output: HookOutput;

    // Check for fast approval first
    const fastApproval = shouldFastApprove(
      hookData.tool_name,
      hookData.tool_input,
      permissionMode
    );
    if (fastApproval) {
      log.info(
        {
          tool: hookData.tool_name,
          decision: fastApproval.hookSpecificOutput.permissionDecision,
          reason: fastApproval.hookSpecificOutput.permissionDecisionReason,
        },
        'Fast approval granted'
      );
      output = fastApproval;
    } else {
      // Check cache for previous decision (only if cache is enabled and not disabled by flag)
      let cachedDecision = null;
      if (config.cache && !noCache) {
        cachedDecision = getCachedDecision(
          hookData.tool_name,
          hookData.tool_input,
          workingDir
        );
      }

      if (cachedDecision) {
        log.info(
          {
            tool: hookData.tool_name,
            decision: cachedDecision.decision,
            reason: cachedDecision.reason,
          },
          'Using cached decision'
        );
        output = {
          hookSpecificOutput: {
            hookEventName: 'PreToolUse',
            permissionDecision: cachedDecision.decision,
            permissionDecisionReason: `${cachedDecision.reason} (cached)`,
          },
        };
      } else {
        // Check if LLM client can be configured
        if (!canConfigureLLMClient()) {
          throw new Error(
            'No authentication method configured. Available options:\n' +
              '1. beyondthehype.dev: Set beyondthehypeApiKey in config (recommended)\n' +
              '2. OpenAI-compatible: Set openaiApiKey/OPENAI_API_KEY or apiKey/ANTHROPIC_API_KEY in config or environment\n' +
              '\nRun `ccb install` to configure authentication interactively.'
          );
        }

        log.debug(
          {
            tool: hookData.tool_name,
            hasApiKey: canConfigureLLMClient(),
          },
          'Querying LLM for decision'
        );

        // Fall back to AI-powered decision making
        const claudeResponse = await queryLLM(
          hookData.tool_name,
          hookData.tool_input,
          permissionMode,
          workingDir
        );

        // Apply permission mode adjustments to LLM decision
        const finalDecision = applyPermissionModeToDecision(
          claudeResponse.decision,
          permissionMode
        );

        output = {
          hookSpecificOutput: {
            hookEventName: 'PreToolUse',
            permissionDecision: finalDecision,
            permissionDecisionReason:
              finalDecision !== claudeResponse.decision
                ? `${claudeResponse.reason} (converted from ${claudeResponse.decision} to ${finalDecision} in ${permissionMode} mode)`
                : claudeResponse.reason,
          },
        };

        // Cache the decision if it's allow or deny (not ask) and cache is enabled and not disabled by flag
        if (config.cache && !noCache && claudeResponse.decision !== 'ask') {
          setCachedDecision(
            hookData.tool_name,
            hookData.tool_input,
            workingDir,
            claudeResponse.decision,
            claudeResponse.reason
          );
        }
      }
    }

    // Log the approval decision if enabled in config
    if (config.log) {
      await logApproval(
        hookData.tool_name,
        hookData.tool_input,
        output.hookSpecificOutput.permissionDecision || 'undefined',
        output.hookSpecificOutput.permissionDecisionReason,
        hookData.session_id
      );
    }

    log.info(
      {
        tool: hookData.tool_name,
        decision: output.hookSpecificOutput.permissionDecision,
        reason: output.hookSpecificOutput.permissionDecisionReason,
        sessionId: hookData.session_id,
      },
      'Final decision made'
    );

    process.stdout.write(JSON.stringify(output));
    process.exit(0);
  } catch (error) {
    log.error(
      { error: error instanceof Error ? error.message : String(error) },
      'Error processing hook input'
    );
    process.stderr.write(`Error processing hook input: ${error}\n`);
    process.exit(1);
  }
}
