import { PageContainer } from '../../components/ui/PageContainer';
import { SectionHeader } from '../../components/ui/SectionHeader';
import { SettingsSection } from './components/SettingsSection';
import { GeneralSettingsForm } from './components/GeneralSettingsForm';
import { VenueConfigurationForm } from './components/VenueConfigurationForm';
import { ZoneConfigurationList } from './components/ZoneConfigurationList';
import { ModelSettingsPanel } from './components/ModelSettingsPanel';
import { NotificationPreferencesPanel } from './components/NotificationPreferencesPanel';
import { SystemInformationPanel } from './components/SystemInformationPanel';

export function SettingsPage() {
  return (
    <PageContainer>
      <SectionHeader as="h1" title="Settings" description="System configuration and preferences." />

      <div className="mt-6 rounded-lg border border-[#E5E7EB] bg-white px-6">
        <SettingsSection title="General" description="Operator identity and display preferences.">
          <GeneralSettingsForm />
        </SettingsSection>

        <SettingsSection title="Venue Configuration" description="Venue identity and rated capacity.">
          <VenueConfigurationForm />
        </SettingsSection>

        <SettingsSection title="Zone Configuration" description="Zones currently provisioned for monitoring.">
          <ZoneConfigurationList />
        </SettingsSection>

        <SettingsSection title="Model Settings" description="Risk model configuration, read-only for now.">
          <ModelSettingsPanel />
        </SettingsSection>

        <SettingsSection title="Notification Preferences" description="Choose what triggers a notification.">
          <NotificationPreferencesPanel />
        </SettingsSection>

        <SettingsSection title="System Information" description="Build and connection status.">
          <SystemInformationPanel />
        </SettingsSection>
      </div>
    </PageContainer>
  );
}
