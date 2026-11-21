import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import { FormField } from '../../../components/forms/FormField';

interface GeneralSettingsValues {
  operatorName: string;
  organization: string;
  timezone: string;
}

export function GeneralSettingsForm() {
  const { register, handleSubmit, formState } = useForm<GeneralSettingsValues>({
    defaultValues: {
      operatorName: 'Control Room Operator',
      organization: 'CrowdShield AI',
      timezone: 'UTC',
    },
  });

  const onSubmit = handleSubmit(() => {
    toast.success('General settings saved');
  });

  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-4">
      <FormField label="Operator name" {...register('operatorName')} />
      <FormField label="Organization" {...register('organization')} />
      <FormField label="Timezone" hint="Used for all timestamps across the dashboard." {...register('timezone')} />
      <div>
        <button
          type="submit"
          disabled={formState.isSubmitting}
          className="rounded-md bg-[#2563EB] px-4 py-2 text-small font-medium text-white hover:bg-[#2563EB]/90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2563EB]"
        >
          Save changes
        </button>
      </div>
    </form>
  );
}
