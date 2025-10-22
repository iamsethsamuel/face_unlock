
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { useThemeColor } from '@/hooks/use-theme-color';
import { router } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { useEffect, useState } from 'react';
import { Button, FlatList, StyleSheet, TextInput, View } from 'react-native';

const USERS_KEY = 'enrolled_users';

export default function WelcomeScreen() {
  const inputColor = useThemeColor({}, 'text');
  const backgroundColor = useThemeColor({}, 'background');
  const [users, setUsers] = useState<string[]>([]);
  const [name, setName] = useState('');

  useEffect(() => {
    const fetchUsers = async () => {
      const usersJson = await SecureStore.getItemAsync(USERS_KEY);
      if (usersJson) {
        setUsers(JSON.parse(usersJson));
      }
    };
    fetchUsers();
  }, []);

  const handleAddUser = () => {
    if (name.trim()) {
      router.push({ pathname: '/enroll', params: { name } });
    }
  };

  return (
    <ThemedView style={styles.container}>
      <ThemedText style={styles.subtitle}>Register a new user:</ThemedText>
      <View style={styles.inputContainer}>
        <TextInput
          style={[styles.input, { color: inputColor, backgroundColor: backgroundColor }]}
          placeholder="Enter your name"
          placeholderTextColor={inputColor === '#11181C' ? '#666' : '#999'}
          value={name}
          onChangeText={setName}
        />
        <Button title="Add" onPress={handleAddUser} />
      </View>
      <ThemedText style={styles.subtitle}>Registered Users:</ThemedText>
      <FlatList
        data={users}
        keyExtractor={(item) => item}
        renderItem={({ item }) => <ThemedText style={styles.userItem}>{item}</ThemedText>}
        ListEmptyComponent={<ThemedText style={styles.emptyText}>No users enrolled yet.</ThemedText>}
      />
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    paddingTop: 60,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  subtitle: {
    fontSize: 18,
    marginTop: 20,
    marginBottom: 10,
  },
  inputContainer: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 10,
    marginRight: 10,
    borderRadius: 5,
  },
  userItem: {
    padding: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  emptyText: {
    fontStyle: 'italic',
    textAlign: 'center',
    marginTop: 20,
  },
});
