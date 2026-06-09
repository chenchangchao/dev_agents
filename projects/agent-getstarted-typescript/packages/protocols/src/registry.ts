export interface AgentProfile {
  id: string;
  name: string;
  capabilities: string[];
  endpoint?: string;
}

export class AgentRegistry {
  private readonly agents = new Map<string, AgentProfile>();

  register(profile: AgentProfile): void {
    this.agents.set(profile.id, profile);
  }

  get(id: string): AgentProfile | undefined {
    return this.agents.get(id);
  }

  findByCapability(capability: string): AgentProfile[] {
    return [...this.agents.values()].filter((agent) => agent.capabilities.includes(capability));
  }

  list(): AgentProfile[] {
    return [...this.agents.values()];
  }
}
