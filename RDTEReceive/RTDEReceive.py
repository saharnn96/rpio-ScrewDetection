import sys
from . import rtde, rtde_config
import csv
from matplotlib.animation import FuncAnimation
from collections import deque
import matplotlib.pyplot as plt


class RTDEReceive:
    def __init__(self, ip_address="192.168.1.100", config_file="record_configuration.xml", frequency=125):
        self.ip_address = ip_address
        self.config_file = config_file
        self.frequency = frequency
        self.connection = rtde.RTDE(self.ip_address)
        self.conf = rtde_config.ConfigFile(self.config_file)
        self.output_names, self.output_types = self.conf.get_recipe("out")

    def connect_to_robot(self):
        self.connection.connect()
        if not self.connection.send_output_setup(self.output_names, self.output_types, frequency=self.frequency):
            raise RuntimeError("Failed to set up output recipe.")

    def start_receiving(self):
        if not self.connection.send_start():
            raise RuntimeError("Failed to start data synchronization.")

    def stop_receiving(self):
        self.connection.send_pause()
        self.connection.disconnect()
        sys.exit()

    def receive_data(self):
        response = self.connection.receive(False)
        if response:
            return self.unpack_data(response)
        return None

    def unpack_data(self, data_object):
        data_packet = {}
        for i in range(len(self.output_names)):
            value = data_object.__dict__[self.output_names[i]]
            data_packet[self.output_names[i]] = value
        return data_packet

    def write_data_to_csv(self, data, filename="robot_data.csv"):
        # Check if the file exists and write the header if it doesn't
        try:
            with open(filename, mode="x", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.output_names)
                writer.writeheader()
        except FileExistsError:
            pass

        # Append the data to the file
        with open(filename, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.output_names)
            writer.writerow(data)

    def live_plot(self, column_name):
        fig, ax = plt.subplots()
        x_data, y_data, smoothed_y_data = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
        line, = ax.plot(x_data, smoothed_y_data, label=f"Smoothed {column_name}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 0.5)
        ax.set_title(f"Live Plot of {column_name} (Smoothed)")
        ax.legend()
        ax.grid(True)

        def update(frame):
            data = self.receive_data()
            if data and column_name in data:
                x_data.append(len(x_data))
                y_data.append(float(data[column_name]))

                # Apply rolling average smoothing
                if len(y_data) >= 5:  # Use a window size of 5
                    smoothed_value = sum(list(y_data)[-5:]) / 5
                else:
                    smoothed_value = sum(y_data) / len(y_data)

                smoothed_y_data.append(smoothed_value)
                line.set_data(range(len(x_data)), smoothed_y_data)
                ax.relim()
                ax.autoscale_view()

        ani = FuncAnimation(fig, update, interval=1000 / self.frequency)
        plt.show()

    def main_loop(self):
        self.connect_to_robot()
        self.start_receiving()
        while True:
            self.live_plot("runtime_state")

            




# Example usage
if __name__ == "__main__":
    rtde_receiver = RTDEReceive(ip_address="192.168.1.100", config_file="record_configuration.xml", frequency=125)
    # rtde_receiver.main_loop()