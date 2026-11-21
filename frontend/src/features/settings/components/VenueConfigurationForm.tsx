import { useForm } from 'react-hook-form';
import { toast } from 'sonner';
import { FormField } from '../../../components/forms/FormField';

interface VenueSettingsValues {
  venueName: string;
  address: string;
  totalCapacity: number;
}

export function VenueConfigurationForm() {
  const { register, handleSubmit, formState } = useForm<VenueSettingsValues>({
    defaultValues: {
      venueName: 'Central Transit Arena',
      address: '1 Arena Way',
      totalCapacity: 48000,
    },
  });

  const onSubmit = handleSubmit(() => {
    toast.success('Venue configuration saved');
  });

  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-4">
      <FormField label="Venue name" {...register('venueName')} />
      <FormField label="Address" {...register('address')} />
      <FormField
        label="Total rated capacity"
        type="number"
        hint="Used to compute occupancy percentages venue-wide."
        {...register('totalCapacity', { valueAsNumber: true })}
      />
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
